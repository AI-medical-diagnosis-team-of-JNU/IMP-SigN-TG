from ops import *

class GAT_Unet(nn.Module):
    def __init__(self, in_channel, dropout, alpha, height, width, batch_size = 3, basic_channel = 4, bilinear=True):
        super(GAT_Unet, self).__init__()
        self.in_channel = in_channel
        self.basic_channel = basic_channel
        self.input_dim = self.basic_channel*int(height*width/64)
        self.hidden_dim = int(height*width/16)
        self.n_classes = height/width
        self.batch_size = batch_size
        self.dropout = dropout
        self.alpha = alpha

        self.Feature_get = Feature_get(self.in_channel, self.basic_channel)
        self.attention_1 = Attention()
        self.attention_2 = Attention()
        self.GAT_1 = GraphAttentionLayer(self.input_dim, self.input_dim, self.dropout, self.alpha)

        factor = 1
        self.up1 = Up(basic_channel*32, basic_channel*8, bilinear)
        self.up2 = Up(basic_channel*16, basic_channel*4 // factor, bilinear)
        self.up3 = Up(basic_channel*8, basic_channel*2 // factor, bilinear)
        self.up4 = Up(basic_channel*4, basic_channel*1 // factor, bilinear)
        self.up5 = Up(basic_channel*2, basic_channel*1 // factor, bilinear)

        self.fc = nn.Sequential(

            nn.Linear(int(height*width/16), 256)
        )

        self.conv_o1 = torch.nn.Conv2d(4,1,kernel_size=1,stride=1)
        self.conv_o2 = torch.nn.Conv2d(1,1,kernel_size=1,stride=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, T0, adj):

        x, f_2, f_3, f_4, f_5, f_6 = self.Feature_get(T0)
        x_ = x
        x = x.view([self.batch_size,-1])

        x = self.GAT_1(x, adj)
        fc = self.fc(x)

        graph_feature = x.view(x_.shape)

        x = self.up1(graph_feature, f_2)
        x = self.up2(x, f_3)
        x = self.up3(x, f_4)
        x = self.up4(x, f_5)
        x = self.up5(x, f_6)

        x = self.conv_o1(x)
        x = self.conv_o2(x)
        out = self.sigmoid(x)
        return out, fc