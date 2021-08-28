class NARTransformerModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, estimate_type, use_feats, t2v_type,
            v_dim, kernel_size, nkernel, device
        ):
        super(NARTransformerModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.estimate_type = estimate_type
        self.use_feats = use_feats
        self.t2v_type = t2v_type
        self.v_dim = v_dim
        self.device = device
        self.use_covariate_var_model = False

        self.kernel_size = kernel_size
        self.nkernel = nkernel

        self.warm_start = self.kernel_size * 5

        if self.use_feats:
            self.embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not -1:
                    if card is not 0:
                        self.embed_feat_layers.append(nn.Embedding(card, emb_size))
                    else:
                        self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

            in_channels = sum([s for (_, s) in self.feats_info.values() if s is not -1])
            self.conv_feats = nn.Conv1d(
                kernel_size=self.kernel_size, stride=1, in_channels=in_channels, out_channels=nkernel,
                bias=False,
                #padding=self.kernel_size//2
            )

        #self.enc_conv_data = nn.Conv1d(
        #    kernel_size=self.kernel_size, stride=1, in_channels=1, out_channels=nkernel,
        #    bias=False,
        #    #padding=self.kernel_size//2
        #)
        #self.dec_conv_data = nn.Conv1d(
        #    kernel_size=self.kernel_size, stride=1, in_channels=1, out_channels=nkernel,
        #    bias=False,
        #    #padding=self.kernel_size//2
        #)
        self.conv_data = nn.Conv1d(
            kernel_size=self.kernel_size, stride=1, in_channels=1, out_channels=nkernel,
            #bias=False,
            #padding=self.kernel_size//2
        )
        self.data_dropout = nn.Dropout(p=0.2)

        if self.use_feats:
            #self.enc_linearMap = nn.Sequential(nn.ReLU(), nn.Linear(2*nkernel, nkernel, bias=False))
            #self.dec_linearMap = nn.Sequential(nn.ReLU(), nn.Linear(2*nkernel, nkernel, bias=False))
            self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(2*nkernel, nkernel, bias=False))
        else:
            #self.enc_linearMap = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, nkernel, bias=False))
            #self.dec_linearMap = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, nkernel, bias=False))
            self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, nkernel, bias=False))
        self.positional = PositionalEncoding(d_model=nkernel)

        enc_input_size = nkernel

        if self.t2v_type:
            if self.t2v_type not in ['local']:  
                self.t_size = sum([1 for (_, s) in self.feats_info.values() if s==-1])
            else:
                self.t_size = 1
            if self.t2v_type in ['mdh_lincomb']:
                self.t2v_layer_list = []
                for i in range(self.t_size):
                    self.t2v_layer_list.append(nn.Linear(1, nkernel))
                self.t2v_layer_list = nn.ModuleList(self.t2v_layer_list)
                if self.t_size > 1:
                    self.t2v_linear =  nn.Linear(self.t_size*nkernel, nkernel)
                else:
                    self.t2v_linear = None
            elif self.t2v_type in ['local', 'mdh_parti', 'idx']:
                self.part_sizes = [nkernel//self.t_size]*self.t_size
                for i in range(nkernel%self.t_size):
                    self.part_sizes[i] += 1
                self.t2v_layer_list = []
                for i in range(len(self.part_sizes)):
                    self.t2v_layer_list.append(nn.Linear(1, self.part_sizes[i]))
                self.t2v_layer_list = nn.ModuleList(self.t2v_layer_list)
                self.t2v_dropout = nn.Dropout(p=0.2)
                self.t2v_linear = None
            #import ipdb ; ipdb.set_trace()

        # important functions / layers of transformer (enc, dec)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_input_size, nhead=4, dropout=0, dim_feedforward=512
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=nkernel, nhead=4, dropout=0, dim_feedforward=512
        )
        self.decoder_mean = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['variance', 'covariance']:
            self.decoder_std = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.linear_mean = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['variance', 'covariance']:
            self.linear_std = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['covariance']:
            self.linear_v = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, self.v_dim))

    def forward(
        self, feats_in, X_in, feats_out, X_out=None, teacher_force=None
    ):

        #X_in = X_in[..., -X_in.shape[1]//5:, :]
        #feats_in = feats_in[..., -feats_in.shape[1]//5:, :]

        mean = X_in.mean(dim=1, keepdim=True)
        #std = X_in.std(dim=1,keepdim=True)
        X_in = (X_in - mean)

        #import ipdb ; ipdb.set_trace()
        if self.use_feats:
            feats_in_merged, feats_out_merged = [], []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_in_ = feats_in[..., i:i+1]
                    feats_in_merged.append(
                        self.embed_feat_layers[i](feats_in_)
                    )
            feats_in_merged = torch.cat(feats_in_merged, dim=2)
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)

            feats_in_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_in.shape[0], self.kernel_size-1, feats_in_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_in_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1],:].clamp(min=0)

        X_in_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        (X_in.shape[0], self.kernel_size-1, X_in.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    X_in
                ], dim=1
            ).transpose(1,2)
        ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1], :]

        if self.use_feats:
            enc_input = self.linearMap(torch.cat([feats_in_embed,X_in_embed],dim=-1)).transpose(0,1)
        else:
            enc_input = self.linearMap(X_in_embed).transpose(0,1)

        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(X_in.shape[1], X_in.shape[0]).unsqueeze(-1)
                t_in = t_in / X_in.shape[1] * 10.
            else:
                t_in = feats_in[..., :, -self.t_size:].transpose(0,1)
            t2v = []
            #if self.t2v_type is 'mdh_lincomb':
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            #import ipdb ; ipdb.set_trace()
            #t2v = torch.cat([t2v[0:1], torch.sin(t2v[1:])], dim=0)
            #enc_input = self.data_dropout(enc_input) + self.t2v_dropout(t2v)
            enc_input = enc_input + self.t2v_dropout(t2v)
        else:
            enc_input = self.positional(enc_input)
        encoder_output = self.encoder(enc_input)

        if self.use_feats:
            feats_out_merged = torch.cat(
                [feats_in_merged[:,-self.warm_start+1:, :],feats_out_merged],
                dim=1
            )
            feats_out_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_in.shape[0], self.kernel_size-1, feats_out_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_out_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)

        #import ipdb ; ipdb.set_trace()
        X_out_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        [X_in.shape[0], self.kernel_size-1, X_in.shape[-1]],
                        dtype=torch.float, device=self.device
                    ),
                    X_in[..., -self.warm_start+1:, :],
                    torch.zeros(
                        [X_in.shape[0], self.dec_len, X_in.shape[-1]],
                        dtype=torch.float, device=self.device
                    )
                ],
                dim=1
            ).transpose(1, 2)
        ).transpose(1, 2)
        #import ipdb ; ipdb.set_trace()
        #X_out_embed = self.dec_conv_data(
        #    torch.zeros(
        #        (X_in.shape[0], self.dec_len+self.kernel_size-1, X_in.shape[2]),
        #        dtype=torch.float, device=self.device
        #    ).transpose(1, 2)
        #).transpose(1, 2)
        #X_out_embed = torch.zeros(
        #    (X_in.shape[0], self.dec_len, X_in.shape[2]),
        #    dtype=torch.float, device=self.device
        #)

        if self.use_feats:
            #dec_input = self.dec_linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
            #dec_input = feats_out_embed.transpose(0,1)
            dec_input = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
        else:
            #dec_input = self.dec_linearMap(X_out_embed).transpose(0,1)
            dec_input = X_out_embed.transpose(0,1)
        #import ipdb ; ipdb.set_trace()
        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], X_in.shape[1]+self.dec_len, dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(self.dec_len, X_in.shape[0]).unsqueeze(-1)
                t_in = t_in / X_in.shape[1] * 10.
            else:
                t_in = feats_out[..., :, -self.t_size:].transpose(0,1)
            t2v = []
            #if self.t2v_type is 'mdh_lincomb':
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            #dec_input = self.data_dropout(dec_input) + self.t2v_dropout(t2v)
            dec_input = dec_input + self.t2v_dropout(t2v)
        else:
            dec_input = self.positional(dec_input, start_idx=X_in.shape[1])
        #import ipdb ; ipdb.set_trace()

        X_out = self.decoder_mean(dec_input, encoder_output).clamp(min=0)
        X_out = X_out.transpose(0,1)
        mean_out = self.linear_mean(X_out)
        #import ipdb ; ipdb.set_trace()
        #mean_out = mean_out*std+mean

        if self.estimate_type in ['variance', 'covariance']:
            X_out = self.decoder_std(dec_input, encoder_output).clamp(min=0)
            X_out = X_out.transpose(0,1)
            std_out = F.softplus(self.linear_std(X_out))
        if self.estimate_type in ['covariance']:
            v_out = self.linear_v(X_out)

        #std_out = self.linear_std(X_out)
        #std_out = F.softplus((std_out*std)/2)
        #std_out = F.softplus(std_out)
        #v_out = self.linear_v(X_out)

        mean_out = mean_out + mean

        if self.estimate_type in ['point']:
            return mean_out[..., -self.dec_len:, :]
        elif self.estimate_type in ['variance']:
            return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :])
        elif self.estimate_type in ['covariance']:
            return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :])
        #return (
        #    mean_out[..., -self.dec_len:, :],
        #    std_out[..., -self.dec_len:, :],
        #    v_out[..., -self.dec_len:, :]
        #)