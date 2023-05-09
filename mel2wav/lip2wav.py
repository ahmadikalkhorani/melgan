import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvNorm3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', activation=torch.nn.ReLU, residual=False, ):
        super(ConvNorm3D, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.residual = residual
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation,
                                      bias=bias)
        self.batched = torch.nn.BatchNorm3d(out_channels)
        self.activation = activation()

        torch.nn.init.xavier_uniform_(
            self.conv3d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        # torch.nn.init.xavier_uniform_(
        #     self.batched.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        # torch.nn.init.xavier_uniform_(
        #     self.activation.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        # pdb.set_trace()
        conv_signal = self.conv3d(signal)

        batched = self.batched(conv_signal)

        if self.residual:
            batched = batched + signal
        activated = self.activation(batched)

        return activated


class Encoder3D(nn.Module):
    """Encoder module:
    - Three 3-d convolution banks
    - Bidirectional LSTM
"""

    def __init__(self, num_out_feat=80, encoder_embedding_dim=384, duration=3.0, fps=25, sr=16000, encoder_n_convolutions=5, num_init_filters=24):
        super(Encoder3D, self).__init__()

        T = int(duration * fps)

        self.out_channel = num_init_filters
        self.in_channel = 3
        convolutions = []

        self.resize = torchvision.transforms.Resize((96, 96))

        for i in range(encoder_n_convolutions):
            if i == 0:
                conv_layer = nn.Sequential(
                    ConvNorm3D(self.in_channel, self.out_channel,
                               kernel_size=5, stride=(1, 2, 2),
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu'),
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=1,
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu', residual=True),
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=1,
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu', residual=True)
                )
                convolutions.append(conv_layer)
            else:
                conv_layer = nn.Sequential(
                    ConvNorm3D(self.in_channel, self.out_channel,
                               kernel_size=3, stride=(1, 2, 2),
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu'),
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=1,
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu', residual=True),
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=1,
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu', residual=True)
                )
                convolutions.append(conv_layer)

            if i == encoder_n_convolutions - 1:
                conv_layer = nn.Sequential(
                    ConvNorm3D(self.out_channel, self.out_channel,
                               kernel_size=3, stride=(1, 3, 3),
                               # padding=int((hparams.encoder_kernel_size - 1) / 2),
                               dilation=1, w_init_gain='relu'),

                )
                convolutions.append(conv_layer)

            self.in_channel = self.out_channel
            self.out_channel *= 2
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(encoder_embedding_dim,
                            num_out_feat, 1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*num_out_feat, num_out_feat)
        self.conv_out = nn.Sequential(
            nn.ConvTranspose1d(
                T, int(((duration*sr)*22050/sr)/256), kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x, input_lengths=None):
        x = x["video"].transpose(-3, -4)
        B, T, C, H, W = x.size()
        x = x.reshape(B*T, C, H, W)
        x = self.resize(x).reshape(B, T, C, 96, 96)
        for conv in self.convolutions:
            x = F.dropout(conv(x), 0.5, self.training)
        # for i in range(len(self.convolutions)):
        # 	if i==0 or i==1 or i ==2:
        # 		with torch.no_grad():
        # 			x = F.dropout(self.convolutions[i](x), 0.5, self.training)
        # 	else:
        # 		x = F.dropout(self.convolutions[i](x), 0.5, self.training)

        x = x.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(
            3).contiguous()  # [bs x 90 x encoder_embedding_dim]
        # print(x.size())
        # pytorch tensor are not reversible, hence the conversion
        # input_lengths = input_lengths.cpu().numpy()
        # x = nn.utils.rnn.pack_padded_sequence(
        # 	x, input_lengths, batch_first=True)

        # self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs = self.fc(outputs)
        outputs = self.conv_out(outputs)

        return outputs.transpose(-1, -2).contiguous()
