本实践使用多层感知器训练（DNN）模型，用于预测手写数字图片。



实践总体过程和步骤如下图



首先导入必要的包

numpy---------->python第三方库，用于进行科学计算

PIL------------> Python Image Library,python第三方图像处理库

matplotlib----->python的绘图库 pyplot:matplotlib的绘图框架

os------------->提供了丰富的方法来处理文件和目录

In [1]
#导入需要的包
import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
import os
from paddle.fluid.dygraph import Linear
2022-03-19 20:39:58,622-INFO: font search path ['/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/afm', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts']
2022-03-19 20:39:59,066-INFO: generated new fontManager
Step1：准备数据。
(1)数据集介绍

MNIST数据集包含60000个训练集和10000测试数据集。分为图片和标签，图片是28*28的像素矩阵，标签为0~9共10个数字。



(2)train_reader和test_reader

paddle.dataset.mnist.train()和test()分别用于获取mnist训练集和测试集

paddle.batch()表示每BATCH_SIZE组成一个batch

paddle.reader.shuffle()表示每次缓存BUF_SIZE个数据项，并在读取一个数据项前进行随机排序。

（3）打印看下数据是什么样的？PaddlePaddle接口提供的数据已经经过了归一化、居中等处理。

In [2]
!mkdir -p /home/aistudio/.cache/paddle/dataset/mnist/
!cp -r /home/aistudio/data/data65/*  /home/aistudio/.cache/paddle/dataset/mnist/
!ls /home/aistudio/.cache/paddle/dataset/mnist/
t10k-images-idx3-ubyte.gz  train-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz  train-labels-idx1-ubyte.gz
In [3]
BUF_SIZE=512
BATCH_SIZE=128
#用于训练的数据提供器，每次从缓存的数据项中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)
#用于训练的数据提供器，每次从缓存的数据项中随机读取批次大小的数据
test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.test(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)
In [4]
#用于打印，查看mnist数据
train_data=paddle.dataset.mnist.train();
sampledata=next(train_data())
print(sampledata)
(array([-1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -0.9764706 , -0.85882354, -0.85882354,
       -0.85882354, -0.01176471,  0.06666672,  0.37254906, -0.79607844,
        0.30196083,  1.        ,  0.9372549 , -0.00392157, -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -0.7647059 , -0.7176471 , -0.26274508,  0.20784318,
        0.33333337,  0.9843137 ,  0.9843137 ,  0.9843137 ,  0.9843137 ,
        0.9843137 ,  0.7647059 ,  0.34901965,  0.9843137 ,  0.8980392 ,
        0.5294118 , -0.4980392 , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -0.6156863 ,  0.8666667 ,
        0.9843137 ,  0.9843137 ,  0.9843137 ,  0.9843137 ,  0.9843137 ,
        0.9843137 ,  0.9843137 ,  0.9843137 ,  0.96862745, -0.27058822,
       -0.35686272, -0.35686272, -0.56078434, -0.69411767, -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -0.85882354,  0.7176471 ,  0.9843137 ,  0.9843137 ,
        0.9843137 ,  0.9843137 ,  0.9843137 ,  0.5529412 ,  0.427451  ,
        0.9372549 ,  0.8901961 , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -0.372549  ,  0.22352946, -0.1607843 ,  0.9843137 ,  0.9843137 ,
        0.60784316, -0.9137255 , -1.        , -0.6627451 ,  0.20784318,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -0.8901961 ,
       -0.99215686,  0.20784318,  0.9843137 , -0.29411763, -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        ,  0.09019613,
        0.9843137 ,  0.4901961 , -0.9843137 , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -0.9137255 ,  0.4901961 ,  0.9843137 ,
       -0.45098037, -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -0.7254902 ,  0.8901961 ,  0.7647059 ,  0.254902  ,
       -0.15294117, -0.99215686, -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -0.36470586,  0.88235295,  0.9843137 ,  0.9843137 , -0.06666666,
       -0.8039216 , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -0.64705884,
        0.45882356,  0.9843137 ,  0.9843137 ,  0.17647064, -0.7882353 ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -0.8745098 , -0.27058822,
        0.9764706 ,  0.9843137 ,  0.4666667 , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        ,  0.9529412 ,  0.9843137 ,
        0.9529412 , -0.4980392 , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -0.6392157 ,  0.0196079 ,
        0.43529415,  0.9843137 ,  0.9843137 ,  0.62352943, -0.9843137 ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -0.69411767,
        0.16078436,  0.79607844,  0.9843137 ,  0.9843137 ,  0.9843137 ,
        0.9607843 ,  0.427451  , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -0.8117647 , -0.10588235,  0.73333335,  0.9843137 ,  0.9843137 ,
        0.9843137 ,  0.9843137 ,  0.5764706 , -0.38823527, -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -0.81960785, -0.4823529 ,  0.67058825,  0.9843137 ,
        0.9843137 ,  0.9843137 ,  0.9843137 ,  0.5529412 , -0.36470586,
       -0.9843137 , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -0.85882354,  0.3411765 ,  0.7176471 ,
        0.9843137 ,  0.9843137 ,  0.9843137 ,  0.9843137 ,  0.5294118 ,
       -0.372549  , -0.92941177, -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -0.5686275 ,  0.34901965,
        0.77254903,  0.9843137 ,  0.9843137 ,  0.9843137 ,  0.9843137 ,
        0.9137255 ,  0.04313731, -0.9137255 , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        ,  0.06666672,  0.9843137 ,  0.9843137 ,  0.9843137 ,
        0.6627451 ,  0.05882359,  0.03529418, -0.8745098 , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        , -1.        ,
       -1.        , -1.        , -1.        , -1.        ], dtype=float32), 5)
Step2.网络配置
以下的代码判断就是定义一个简单的多层感知器，一共有三层，两个大小为100的隐层和一个大小为10的输出层，因为MNIST数据集是手写0到9的灰度图像，类别有10个，所以最后的输出大小是10。最后输出层的激活函数是Softmax，所以最后的输出层相当于一个分类器。加上一个输入层的话，多层感知器的结构是：输入层-->>隐层-->>隐层-->>输出层。



In [5]
# 定义多层感知器 
#动态图定义多层感知器
class multilayer_perceptron(fluid.dygraph.Layer):
    def __init__(self):
        super(multilayer_perceptron,self).__init__()
        self.fc1 = Linear(input_dim=28*28, output_dim=100, act='relu')
        self.fc2 = Linear(input_dim=100, output_dim=100, act='relu')
        self.fc3 = Linear(input_dim=100, output_dim=10,act="softmax")
    def forward(self,input_):
        x = fluid.layers.reshape(input_, [input_.shape[0], -1])
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)
        return y
展示模型训练曲线

In [6]
all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]


#绘制训练过程
def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()


def draw_process(title,color,iters,data,label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data,color=color,label=label) 
    plt.legend()
    plt.grid()
    plt.show()
（4）训练并保存模型

训练需要有一个训练程序和一些必要参数，并构建了一个获取训练过程中测试误差的函数。必要参数有executor,program,reader,feeder,fetch_list。

In [7]
#用动态图进行训练
all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

best_test_acc = 0.0


with fluid.dygraph.guard():
    model=multilayer_perceptron() #模型实例化
    model.train() #训练模式
    opt=fluid.optimizer.Adam(learning_rate=fluid.dygraph.ExponentialDecay(
              learning_rate=0.005,
              decay_steps=4000,
              decay_rate=0.1,
              staircase=True), parameter_list=model.parameters())
    
  
    epochs_num=30 #迭代次数
    
    for pass_num in range(epochs_num):
        lr = opt.current_step_lr()
        print("learning-rate:", lr)
        
        for batch_id,data in enumerate(train_reader()):
            
            images=np.array([x[0].reshape(1,28,28) for x in data],np.float32)
            
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]
            
            image=fluid.dygraph.to_variable(images)
            label=fluid.dygraph.to_variable(labels)
            predict=model(image)#预测
            #print(predict)
            loss=fluid.layers.cross_entropy(predict,label)
            avg_loss=fluid.layers.mean(loss)#获取loss值
            
            acc=fluid.layers.accuracy(predict,label)#计算精度
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()

            all_train_iter=all_train_iter+256
            all_train_iters.append(all_train_iter)
            all_train_costs.append(loss.numpy()[0])
            all_train_accs.append(acc.numpy()[0])
            
            
            if batch_id!=0 and batch_id%50==0:
                print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
        
        
        with fluid.dygraph.guard():
            accs = []
            model.eval()#评估模式
            for batch_id,data in enumerate(test_reader()):#测试集
                images=np.array([x[0].reshape(1,28,28) for x in data],np.float32)
                labels = np.array([x[1] for x in data]).astype('int64')
                labels = labels[:, np.newaxis]
            
                image=fluid.dygraph.to_variable(images)
                label=fluid.dygraph.to_variable(labels)
            
                predict=model(image)#预测
                acc=fluid.layers.accuracy(predict,label)
                accs.append(acc.numpy()[0])
                avg_acc = np.mean(accs)
           
           
            if avg_acc >= best_test_acc:
                best_test_acc = avg_acc
                if pass_num > 10:
                    fluid.save_dygraph(model.state_dict(),'./work/{}'.format(pass_num))#保存模型
            
            print('Test:%d, Accuracy:%0.5f, Best: %0.5f'%  (pass_num, avg_acc, best_test_acc))

            
    #fluid.save_dygraph(model.state_dict(),'./work/fashion_mnist_epoch{}'.format(epochs_num))#保存模型


print('训练模型保存完成！')
print("best_test_acc", best_test_acc)
draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")  
draw_process("trainning loss","red",all_train_iters,all_train_costs,"trainning loss")
draw_process("trainning acc","green",all_train_iters,all_train_accs,"trainning acc")  
W0319 20:40:00.401706   119 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.0
W0319 20:40:00.407415   119 device_context.cc:260] device: 0, cuDNN Version: 7.6.
learning-rate: 0.005
train_pass:0,batch_id:50,train_loss:[0.32788998],train_acc:[0.8984375]
train_pass:0,batch_id:100,train_loss:[0.58632123],train_acc:[0.78125]
train_pass:0,batch_id:150,train_loss:[0.512415],train_acc:[0.8359375]
train_pass:0,batch_id:200,train_loss:[0.27548963],train_acc:[0.921875]
train_pass:0,batch_id:250,train_loss:[0.45774978],train_acc:[0.8671875]
train_pass:0,batch_id:300,train_loss:[0.24523564],train_acc:[0.9140625]
train_pass:0,batch_id:350,train_loss:[0.20310913],train_acc:[0.9375]
train_pass:0,batch_id:400,train_loss:[0.2542602],train_acc:[0.921875]
train_pass:0,batch_id:450,train_loss:[0.23372617],train_acc:[0.9140625]
Test:0, Accuracy:0.92573, Best: 0.92573
learning-rate: 0.005
train_pass:1,batch_id:50,train_loss:[0.12094763],train_acc:[0.953125]
train_pass:1,batch_id:100,train_loss:[0.25674435],train_acc:[0.921875]
train_pass:1,batch_id:150,train_loss:[0.15837233],train_acc:[0.9140625]
train_pass:1,batch_id:200,train_loss:[0.19065191],train_acc:[0.9453125]
train_pass:1,batch_id:250,train_loss:[0.24840106],train_acc:[0.8984375]
train_pass:1,batch_id:300,train_loss:[0.18741834],train_acc:[0.9453125]
train_pass:1,batch_id:350,train_loss:[0.12199847],train_acc:[0.9609375]
train_pass:1,batch_id:400,train_loss:[0.20142758],train_acc:[0.9296875]
train_pass:1,batch_id:450,train_loss:[0.12759553],train_acc:[0.96875]
Test:1, Accuracy:0.93730, Best: 0.93730
learning-rate: 0.005
train_pass:2,batch_id:50,train_loss:[0.20146237],train_acc:[0.9296875]
train_pass:2,batch_id:100,train_loss:[0.25469735],train_acc:[0.9453125]
train_pass:2,batch_id:150,train_loss:[0.20178755],train_acc:[0.9609375]
train_pass:2,batch_id:200,train_loss:[0.10581377],train_acc:[0.96875]
train_pass:2,batch_id:250,train_loss:[0.20360509],train_acc:[0.9140625]
train_pass:2,batch_id:300,train_loss:[0.15507025],train_acc:[0.9609375]
train_pass:2,batch_id:350,train_loss:[0.09223197],train_acc:[0.96875]
train_pass:2,batch_id:400,train_loss:[0.17168733],train_acc:[0.953125]
train_pass:2,batch_id:450,train_loss:[0.08207137],train_acc:[0.96875]
Test:2, Accuracy:0.94986, Best: 0.94986
learning-rate: 0.005
train_pass:3,batch_id:50,train_loss:[0.12680659],train_acc:[0.953125]
train_pass:3,batch_id:100,train_loss:[0.14775841],train_acc:[0.96875]
train_pass:3,batch_id:150,train_loss:[0.11668487],train_acc:[0.9609375]
train_pass:3,batch_id:200,train_loss:[0.06463864],train_acc:[0.9765625]
train_pass:3,batch_id:250,train_loss:[0.2376196],train_acc:[0.9140625]
train_pass:3,batch_id:300,train_loss:[0.21831512],train_acc:[0.9296875]
train_pass:3,batch_id:350,train_loss:[0.07714231],train_acc:[0.9765625]
train_pass:3,batch_id:400,train_loss:[0.12881309],train_acc:[0.9453125]
train_pass:3,batch_id:450,train_loss:[0.08552274],train_acc:[0.9765625]
Test:3, Accuracy:0.95728, Best: 0.95728
learning-rate: 0.005
train_pass:4,batch_id:50,train_loss:[0.04346458],train_acc:[0.984375]
train_pass:4,batch_id:100,train_loss:[0.17580502],train_acc:[0.9453125]
train_pass:4,batch_id:150,train_loss:[0.34360668],train_acc:[0.9296875]
train_pass:4,batch_id:200,train_loss:[0.1414224],train_acc:[0.953125]
train_pass:4,batch_id:250,train_loss:[0.17935738],train_acc:[0.953125]
train_pass:4,batch_id:300,train_loss:[0.13710229],train_acc:[0.9609375]
train_pass:4,batch_id:350,train_loss:[0.06654441],train_acc:[0.984375]
train_pass:4,batch_id:400,train_loss:[0.14366467],train_acc:[0.96875]
train_pass:4,batch_id:450,train_loss:[0.12757161],train_acc:[0.953125]
Test:4, Accuracy:0.95639, Best: 0.95728
learning-rate: 0.005
train_pass:5,batch_id:50,train_loss:[0.11645316],train_acc:[0.96875]
train_pass:5,batch_id:100,train_loss:[0.1981837],train_acc:[0.9375]
train_pass:5,batch_id:150,train_loss:[0.04144334],train_acc:[0.9765625]
train_pass:5,batch_id:200,train_loss:[0.0419329],train_acc:[0.9921875]
train_pass:5,batch_id:250,train_loss:[0.1696506],train_acc:[0.9453125]
train_pass:5,batch_id:300,train_loss:[0.07438713],train_acc:[0.9765625]
train_pass:5,batch_id:350,train_loss:[0.0684984],train_acc:[0.9765625]
train_pass:5,batch_id:400,train_loss:[0.07266394],train_acc:[0.9765625]
train_pass:5,batch_id:450,train_loss:[0.14554231],train_acc:[0.9375]
Test:5, Accuracy:0.95926, Best: 0.95926
learning-rate: 0.005
train_pass:6,batch_id:50,train_loss:[0.04855406],train_acc:[0.9921875]
train_pass:6,batch_id:100,train_loss:[0.08741274],train_acc:[0.9765625]
train_pass:6,batch_id:150,train_loss:[0.11818501],train_acc:[0.953125]
train_pass:6,batch_id:200,train_loss:[0.09544888],train_acc:[0.96875]
train_pass:6,batch_id:250,train_loss:[0.07937448],train_acc:[0.9765625]
train_pass:6,batch_id:300,train_loss:[0.01845841],train_acc:[1.]
train_pass:6,batch_id:350,train_loss:[0.0658698],train_acc:[0.9765625]
train_pass:6,batch_id:400,train_loss:[0.09234126],train_acc:[0.9609375]
train_pass:6,batch_id:450,train_loss:[0.09969538],train_acc:[0.9609375]
Test:6, Accuracy:0.95411, Best: 0.95926
learning-rate: 0.005
train_pass:7,batch_id:50,train_loss:[0.07159689],train_acc:[0.9765625]
train_pass:7,batch_id:100,train_loss:[0.05553303],train_acc:[0.96875]
train_pass:7,batch_id:150,train_loss:[0.11395585],train_acc:[0.953125]
train_pass:7,batch_id:200,train_loss:[0.08143274],train_acc:[0.984375]
train_pass:7,batch_id:250,train_loss:[0.08705543],train_acc:[0.9609375]
train_pass:7,batch_id:300,train_loss:[0.09455456],train_acc:[0.9765625]
train_pass:7,batch_id:350,train_loss:[0.19420432],train_acc:[0.9296875]
train_pass:7,batch_id:400,train_loss:[0.07494349],train_acc:[0.96875]
train_pass:7,batch_id:450,train_loss:[0.06323607],train_acc:[0.9765625]
Test:7, Accuracy:0.95362, Best: 0.95926
learning-rate: 0.005
train_pass:8,batch_id:50,train_loss:[0.05357043],train_acc:[0.9921875]
train_pass:8,batch_id:100,train_loss:[0.14896885],train_acc:[0.96875]
train_pass:8,batch_id:150,train_loss:[0.2721033],train_acc:[0.9453125]
train_pass:8,batch_id:200,train_loss:[0.0314666],train_acc:[0.984375]
train_pass:8,batch_id:250,train_loss:[0.21879229],train_acc:[0.9375]
train_pass:8,batch_id:300,train_loss:[0.06569576],train_acc:[0.9765625]
train_pass:8,batch_id:350,train_loss:[0.10325915],train_acc:[0.96875]
train_pass:8,batch_id:400,train_loss:[0.10301319],train_acc:[0.984375]
train_pass:8,batch_id:450,train_loss:[0.03785065],train_acc:[0.9921875]
Test:8, Accuracy:0.96974, Best: 0.96974
learning-rate: 0.0005
train_pass:9,batch_id:50,train_loss:[0.01718002],train_acc:[1.]
train_pass:9,batch_id:100,train_loss:[0.03255708],train_acc:[0.984375]
train_pass:9,batch_id:150,train_loss:[0.01126656],train_acc:[0.9921875]
train_pass:9,batch_id:200,train_loss:[0.07518683],train_acc:[0.9765625]
train_pass:9,batch_id:250,train_loss:[0.02267333],train_acc:[0.9921875]
train_pass:9,batch_id:300,train_loss:[0.03326255],train_acc:[0.9765625]
train_pass:9,batch_id:350,train_loss:[0.05405357],train_acc:[0.984375]
train_pass:9,batch_id:400,train_loss:[0.12282136],train_acc:[0.96875]
train_pass:9,batch_id:450,train_loss:[0.03249552],train_acc:[0.984375]
Test:9, Accuracy:0.97053, Best: 0.97053
learning-rate: 0.0005
train_pass:10,batch_id:50,train_loss:[0.02077684],train_acc:[0.9921875]
train_pass:10,batch_id:100,train_loss:[0.02343858],train_acc:[0.9921875]
train_pass:10,batch_id:150,train_loss:[0.04290292],train_acc:[0.984375]
train_pass:10,batch_id:200,train_loss:[0.04629882],train_acc:[0.9765625]
train_pass:10,batch_id:250,train_loss:[0.02541009],train_acc:[0.9921875]
train_pass:10,batch_id:300,train_loss:[0.08386145],train_acc:[0.96875]
train_pass:10,batch_id:350,train_loss:[0.01388561],train_acc:[1.]
train_pass:10,batch_id:400,train_loss:[0.01996492],train_acc:[0.984375]
train_pass:10,batch_id:450,train_loss:[0.05885116],train_acc:[0.9765625]
Test:10, Accuracy:0.97142, Best: 0.97142
learning-rate: 0.0005
train_pass:11,batch_id:50,train_loss:[0.0074518],train_acc:[1.]
train_pass:11,batch_id:100,train_loss:[0.0422457],train_acc:[0.9921875]
train_pass:11,batch_id:150,train_loss:[0.05741469],train_acc:[0.984375]
train_pass:11,batch_id:200,train_loss:[0.06551164],train_acc:[0.984375]
train_pass:11,batch_id:250,train_loss:[0.02040645],train_acc:[1.]
train_pass:11,batch_id:300,train_loss:[0.035988],train_acc:[0.9921875]
train_pass:11,batch_id:350,train_loss:[0.02147057],train_acc:[1.]
train_pass:11,batch_id:400,train_loss:[0.08111568],train_acc:[0.96875]
train_pass:11,batch_id:450,train_loss:[0.05413925],train_acc:[0.9765625]
Test:11, Accuracy:0.97093, Best: 0.97142
learning-rate: 0.0005
train_pass:12,batch_id:50,train_loss:[0.00927204],train_acc:[1.]
train_pass:12,batch_id:100,train_loss:[0.06384139],train_acc:[0.984375]
train_pass:12,batch_id:150,train_loss:[0.0342586],train_acc:[0.9921875]
train_pass:12,batch_id:200,train_loss:[0.01215122],train_acc:[1.]
train_pass:12,batch_id:250,train_loss:[0.02867358],train_acc:[0.9921875]
train_pass:12,batch_id:300,train_loss:[0.0517654],train_acc:[0.984375]
train_pass:12,batch_id:350,train_loss:[0.0411564],train_acc:[0.9921875]
train_pass:12,batch_id:400,train_loss:[0.01694953],train_acc:[0.9921875]
train_pass:12,batch_id:450,train_loss:[0.04567231],train_acc:[0.984375]
Test:12, Accuracy:0.97023, Best: 0.97142
learning-rate: 0.0005
train_pass:13,batch_id:50,train_loss:[0.01057869],train_acc:[1.]
train_pass:13,batch_id:100,train_loss:[0.04874147],train_acc:[0.984375]
train_pass:13,batch_id:150,train_loss:[0.04510427],train_acc:[0.9921875]
train_pass:13,batch_id:200,train_loss:[0.0654908],train_acc:[0.984375]
train_pass:13,batch_id:250,train_loss:[0.03514893],train_acc:[0.984375]
train_pass:13,batch_id:300,train_loss:[0.03248284],train_acc:[0.9921875]
train_pass:13,batch_id:350,train_loss:[0.0213183],train_acc:[1.]
train_pass:13,batch_id:400,train_loss:[0.01001661],train_acc:[1.]
train_pass:13,batch_id:450,train_loss:[0.04681352],train_acc:[0.984375]
Test:13, Accuracy:0.97201, Best: 0.97201
learning-rate: 0.0005
train_pass:14,batch_id:50,train_loss:[0.00356455],train_acc:[1.]
train_pass:14,batch_id:100,train_loss:[0.02793964],train_acc:[0.9921875]
train_pass:14,batch_id:150,train_loss:[0.03597644],train_acc:[0.984375]
train_pass:14,batch_id:200,train_loss:[0.06638589],train_acc:[0.9921875]
train_pass:14,batch_id:250,train_loss:[0.02710332],train_acc:[0.9921875]
train_pass:14,batch_id:300,train_loss:[0.00774194],train_acc:[1.]
train_pass:14,batch_id:350,train_loss:[0.01373952],train_acc:[1.]
train_pass:14,batch_id:400,train_loss:[0.0096017],train_acc:[1.]
train_pass:14,batch_id:450,train_loss:[0.0177823],train_acc:[0.9921875]
Test:14, Accuracy:0.97300, Best: 0.97300
learning-rate: 0.0005
train_pass:15,batch_id:50,train_loss:[0.0138055],train_acc:[1.]
train_pass:15,batch_id:100,train_loss:[0.01367957],train_acc:[1.]
train_pass:15,batch_id:150,train_loss:[0.01859053],train_acc:[1.]
train_pass:15,batch_id:200,train_loss:[0.02705408],train_acc:[0.9921875]
train_pass:15,batch_id:250,train_loss:[0.03816488],train_acc:[0.984375]
train_pass:15,batch_id:300,train_loss:[0.03346791],train_acc:[0.9921875]
train_pass:15,batch_id:350,train_loss:[0.02051868],train_acc:[0.9921875]
train_pass:15,batch_id:400,train_loss:[0.01128136],train_acc:[1.]
train_pass:15,batch_id:450,train_loss:[0.03365185],train_acc:[0.984375]
Test:15, Accuracy:0.97221, Best: 0.97300
learning-rate: 0.0005
train_pass:16,batch_id:50,train_loss:[0.00425714],train_acc:[1.]
train_pass:16,batch_id:100,train_loss:[0.04003416],train_acc:[0.984375]
train_pass:16,batch_id:150,train_loss:[0.0500649],train_acc:[0.984375]
train_pass:16,batch_id:200,train_loss:[0.02338087],train_acc:[0.9921875]
train_pass:16,batch_id:250,train_loss:[0.02793352],train_acc:[0.9921875]
train_pass:16,batch_id:300,train_loss:[0.02761606],train_acc:[0.9921875]
train_pass:16,batch_id:350,train_loss:[0.01250369],train_acc:[1.]
train_pass:16,batch_id:400,train_loss:[0.05914038],train_acc:[0.984375]
train_pass:16,batch_id:450,train_loss:[0.0178368],train_acc:[0.9921875]
Test:16, Accuracy:0.97231, Best: 0.97300
learning-rate: 0.0005
train_pass:17,batch_id:50,train_loss:[0.00923152],train_acc:[1.]
train_pass:17,batch_id:100,train_loss:[0.023677],train_acc:[0.9921875]
train_pass:17,batch_id:150,train_loss:[0.02001486],train_acc:[0.9921875]
train_pass:17,batch_id:200,train_loss:[0.02702829],train_acc:[0.984375]
train_pass:17,batch_id:250,train_loss:[0.00738989],train_acc:[1.]
train_pass:17,batch_id:300,train_loss:[0.00785155],train_acc:[1.]
train_pass:17,batch_id:350,train_loss:[0.0368622],train_acc:[0.9921875]
train_pass:17,batch_id:400,train_loss:[0.02024068],train_acc:[0.9921875]
train_pass:17,batch_id:450,train_loss:[0.0204553],train_acc:[0.9921875]
Test:17, Accuracy:0.97468, Best: 0.97468
learning-rate: 5.0000002e-05
train_pass:18,batch_id:50,train_loss:[0.00308301],train_acc:[1.]
train_pass:18,batch_id:100,train_loss:[0.04301485],train_acc:[0.984375]
train_pass:18,batch_id:150,train_loss:[0.03514434],train_acc:[0.984375]
train_pass:18,batch_id:200,train_loss:[0.01876639],train_acc:[0.9921875]
train_pass:18,batch_id:250,train_loss:[0.00383943],train_acc:[1.]
train_pass:18,batch_id:300,train_loss:[0.00479014],train_acc:[1.]
train_pass:18,batch_id:350,train_loss:[0.02427222],train_acc:[0.9921875]
train_pass:18,batch_id:400,train_loss:[0.08092349],train_acc:[0.9765625]
train_pass:18,batch_id:450,train_loss:[0.01913755],train_acc:[0.9921875]
Test:18, Accuracy:0.97389, Best: 0.97468
learning-rate: 5.0000002e-05
train_pass:19,batch_id:50,train_loss:[0.02401943],train_acc:[0.9921875]
train_pass:19,batch_id:100,train_loss:[0.0124525],train_acc:[1.]
train_pass:19,batch_id:150,train_loss:[0.03025256],train_acc:[0.9921875]
train_pass:19,batch_id:200,train_loss:[0.02819722],train_acc:[0.984375]
train_pass:19,batch_id:250,train_loss:[0.03194476],train_acc:[0.9921875]
train_pass:19,batch_id:300,train_loss:[0.03910929],train_acc:[0.984375]
train_pass:19,batch_id:350,train_loss:[0.00977404],train_acc:[1.]
train_pass:19,batch_id:400,train_loss:[0.01277885],train_acc:[0.9921875]
train_pass:19,batch_id:450,train_loss:[0.0128328],train_acc:[1.]
Test:19, Accuracy:0.97271, Best: 0.97468
learning-rate: 5.0000002e-05
train_pass:20,batch_id:50,train_loss:[0.0101604],train_acc:[1.]
train_pass:20,batch_id:100,train_loss:[0.0100423],train_acc:[1.]
train_pass:20,batch_id:150,train_loss:[0.00647692],train_acc:[1.]
train_pass:20,batch_id:200,train_loss:[0.0165293],train_acc:[0.9921875]
train_pass:20,batch_id:250,train_loss:[0.03800295],train_acc:[0.9921875]
train_pass:20,batch_id:300,train_loss:[0.02575842],train_acc:[0.9921875]
train_pass:20,batch_id:350,train_loss:[0.02448996],train_acc:[0.9921875]
train_pass:20,batch_id:400,train_loss:[0.0244339],train_acc:[0.9765625]
train_pass:20,batch_id:450,train_loss:[0.01982289],train_acc:[0.9921875]
Test:20, Accuracy:0.97468, Best: 0.97468
learning-rate: 5.0000002e-05
train_pass:21,batch_id:50,train_loss:[0.00814995],train_acc:[1.]
train_pass:21,batch_id:100,train_loss:[0.01613403],train_acc:[1.]
train_pass:21,batch_id:150,train_loss:[0.01826155],train_acc:[0.9921875]
train_pass:21,batch_id:200,train_loss:[0.02214332],train_acc:[0.984375]
train_pass:21,batch_id:250,train_loss:[0.00991445],train_acc:[1.]
train_pass:21,batch_id:300,train_loss:[0.01972238],train_acc:[0.9921875]
train_pass:21,batch_id:350,train_loss:[0.01378801],train_acc:[1.]
train_pass:21,batch_id:400,train_loss:[0.01698218],train_acc:[0.984375]
train_pass:21,batch_id:450,train_loss:[0.01796945],train_acc:[0.9921875]
Test:21, Accuracy:0.97468, Best: 0.97468
learning-rate: 5.0000002e-05
train_pass:22,batch_id:50,train_loss:[0.00801724],train_acc:[1.]
train_pass:22,batch_id:100,train_loss:[0.03283789],train_acc:[0.9921875]
train_pass:22,batch_id:150,train_loss:[0.01835611],train_acc:[0.9921875]
train_pass:22,batch_id:200,train_loss:[0.00635274],train_acc:[1.]
train_pass:22,batch_id:250,train_loss:[0.00351447],train_acc:[1.]
train_pass:22,batch_id:300,train_loss:[0.00642632],train_acc:[1.]
train_pass:22,batch_id:350,train_loss:[0.00759022],train_acc:[1.]
train_pass:22,batch_id:400,train_loss:[0.03598299],train_acc:[0.984375]
train_pass:22,batch_id:450,train_loss:[0.02222464],train_acc:[0.9921875]
Test:22, Accuracy:0.97458, Best: 0.97468
learning-rate: 5.0000002e-05
train_pass:23,batch_id:50,train_loss:[0.02383998],train_acc:[0.9921875]
train_pass:23,batch_id:100,train_loss:[0.0209738],train_acc:[0.9921875]
train_pass:23,batch_id:150,train_loss:[0.05292988],train_acc:[0.984375]
train_pass:23,batch_id:200,train_loss:[0.04081687],train_acc:[0.9921875]
train_pass:23,batch_id:250,train_loss:[0.00918057],train_acc:[1.]
train_pass:23,batch_id:300,train_loss:[0.01313567],train_acc:[1.]
train_pass:23,batch_id:350,train_loss:[0.02947595],train_acc:[0.9921875]
train_pass:23,batch_id:400,train_loss:[0.00842918],train_acc:[1.]
train_pass:23,batch_id:450,train_loss:[0.01454694],train_acc:[0.9921875]
Test:23, Accuracy:0.97449, Best: 0.97468
learning-rate: 5.0000002e-05
train_pass:24,batch_id:50,train_loss:[0.01851984],train_acc:[0.9921875]
train_pass:24,batch_id:100,train_loss:[0.01431876],train_acc:[1.]
train_pass:24,batch_id:150,train_loss:[0.00640907],train_acc:[1.]
train_pass:24,batch_id:200,train_loss:[0.05348068],train_acc:[0.9765625]
train_pass:24,batch_id:250,train_loss:[0.00855017],train_acc:[1.]
train_pass:24,batch_id:300,train_loss:[0.01299841],train_acc:[0.9921875]
train_pass:24,batch_id:350,train_loss:[0.02156046],train_acc:[1.]
train_pass:24,batch_id:400,train_loss:[0.03966337],train_acc:[0.984375]
train_pass:24,batch_id:450,train_loss:[0.05205683],train_acc:[0.9765625]
Test:24, Accuracy:0.97429, Best: 0.97468
learning-rate: 5.0000002e-05
train_pass:25,batch_id:50,train_loss:[0.00827933],train_acc:[1.]
train_pass:25,batch_id:100,train_loss:[0.03720404],train_acc:[0.984375]
train_pass:25,batch_id:150,train_loss:[0.01134678],train_acc:[1.]
train_pass:25,batch_id:200,train_loss:[0.02547757],train_acc:[0.984375]
train_pass:25,batch_id:250,train_loss:[0.02966347],train_acc:[0.9921875]
train_pass:25,batch_id:300,train_loss:[0.04092827],train_acc:[0.984375]
train_pass:25,batch_id:350,train_loss:[0.01614666],train_acc:[0.9921875]
train_pass:25,batch_id:400,train_loss:[0.04311441],train_acc:[0.9765625]
train_pass:25,batch_id:450,train_loss:[0.03364436],train_acc:[0.984375]
Test:25, Accuracy:0.97290, Best: 0.97468
learning-rate: 5.0000003e-06
train_pass:26,batch_id:50,train_loss:[0.00967989],train_acc:[1.]
train_pass:26,batch_id:100,train_loss:[0.01881029],train_acc:[1.]
train_pass:26,batch_id:150,train_loss:[0.00958414],train_acc:[1.]
train_pass:26,batch_id:200,train_loss:[0.0266603],train_acc:[0.984375]
train_pass:26,batch_id:250,train_loss:[0.00597296],train_acc:[1.]
train_pass:26,batch_id:300,train_loss:[0.01066247],train_acc:[1.]
train_pass:26,batch_id:350,train_loss:[0.00491738],train_acc:[1.]
train_pass:26,batch_id:400,train_loss:[0.00608987],train_acc:[1.]
train_pass:26,batch_id:450,train_loss:[0.02832444],train_acc:[0.984375]
Test:26, Accuracy:0.97369, Best: 0.97468
learning-rate: 5.0000003e-06
train_pass:27,batch_id:50,train_loss:[0.01713889],train_acc:[0.9921875]
train_pass:27,batch_id:100,train_loss:[0.01194714],train_acc:[1.]
train_pass:27,batch_id:150,train_loss:[0.01455088],train_acc:[1.]
train_pass:27,batch_id:200,train_loss:[0.00941454],train_acc:[1.]
train_pass:27,batch_id:250,train_loss:[0.00900551],train_acc:[1.]
train_pass:27,batch_id:300,train_loss:[0.00963983],train_acc:[1.]
train_pass:27,batch_id:350,train_loss:[0.00614853],train_acc:[1.]
train_pass:27,batch_id:400,train_loss:[0.05790813],train_acc:[0.984375]
train_pass:27,batch_id:450,train_loss:[0.02287354],train_acc:[0.9921875]
Test:27, Accuracy:0.97449, Best: 0.97468
learning-rate: 5.0000003e-06
train_pass:28,batch_id:50,train_loss:[0.00964481],train_acc:[0.9921875]
train_pass:28,batch_id:100,train_loss:[0.01159403],train_acc:[1.]
train_pass:28,batch_id:150,train_loss:[0.00598307],train_acc:[1.]
train_pass:28,batch_id:200,train_loss:[0.01775591],train_acc:[0.9921875]
train_pass:28,batch_id:250,train_loss:[0.00626679],train_acc:[1.]
train_pass:28,batch_id:300,train_loss:[0.02423741],train_acc:[0.9921875]
train_pass:28,batch_id:350,train_loss:[0.01134321],train_acc:[1.]
train_pass:28,batch_id:400,train_loss:[0.04340545],train_acc:[0.9765625]
train_pass:28,batch_id:450,train_loss:[0.02681123],train_acc:[0.9921875]
Test:28, Accuracy:0.97508, Best: 0.97508
learning-rate: 5.0000003e-06
train_pass:29,batch_id:50,train_loss:[0.02037274],train_acc:[0.9921875]
train_pass:29,batch_id:100,train_loss:[0.0123938],train_acc:[1.]
train_pass:29,batch_id:150,train_loss:[0.00227636],train_acc:[1.]
train_pass:29,batch_id:200,train_loss:[0.02213205],train_acc:[0.984375]
train_pass:29,batch_id:250,train_loss:[0.01048672],train_acc:[1.]
train_pass:29,batch_id:300,train_loss:[0.00523458],train_acc:[1.]
train_pass:29,batch_id:350,train_loss:[0.01694404],train_acc:[1.]
train_pass:29,batch_id:400,train_loss:[0.04442834],train_acc:[0.9921875]
train_pass:29,batch_id:450,train_loss:[0.02608357],train_acc:[0.984375]
Test:29, Accuracy:0.97508, Best: 0.97508
训练模型保存完成！
best_test_acc 0.9750791

<Figure size 432x288 with 1 Axes>

<Figure size 432x288 with 1 Axes>

<Figure size 432x288 with 1 Axes>
Step5.模型预测
（1）图片预处理

在预测之前，要对图像进行预处理。

首先进行灰度化，然后压缩图像大小为28*28，接着将图像转换成一维向量，最后再对一维向量进行归一化处理。

In [8]
def load_image(file):
    im = Image.open(file).convert('L')                        #将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
    im = im.resize((28, 28), Image.ANTIALIAS)                 #resize image with high-quality 图像大小为28*28
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)#返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    # print(im)
    im = im / 255.0 * 2.0 - 1.0                               #归一化到【-1~1】之间
    return im
（2）使用Matplotlib工具显示这张图像并预测。

In [10]

infer_path='/home/aistudio/data/data2304/infer_3.png'
img = Image.open(infer_path)
plt.imshow(img)   #根据数组绘制图像
plt.show()        #显示图像
label_list = [
    "0", "1", "2", "3", "4", "5", "6", "7","8", "9","10"
    ]


#构建预测动态图过程
model_path = './work/24'
with fluid.dygraph.guard():
    model=multilayer_perceptron()#模型实例化
    model_dict,_=fluid.load_dygraph(model_path)
    model.load_dict(model_dict)#加载模型参数
    model.eval()#评估模式
    infer_img = load_image(infer_path)
    infer_img=np.array(infer_img).astype('float32')
    infer_img=infer_img[np.newaxis,:, : ,:]
    infer_img = fluid.dygraph.to_variable(infer_img)
    result=model(infer_img)
   
    print("infer results: %s" % label_list[np.argmax(result.numpy())])

<Figure size 432x288 with 1 Axes>
infer results: 3
In [ ]
In [ ]
In [ ]
In [ ]
In [ ]
In [ ]
代码解释
In [ ]
In [ ]
In [ ]
