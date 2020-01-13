import tensorflow as tf
import tensorflow.contrib.layers as tcl
import scipy.io as sio
import numpy as np
from PIL import Image


H = 64
W = 64
C = 3
batch_size = 128
epclion = 1e-14


def deconv2(x, shape,stride,outshape):
    fiters = tf.get_variable('fiter',shape=shape,initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable('b',shape=[shape[-2]],initializer= tf.constant_initializer ([0]))
    return tf.nn.conv2d_transpose(x,fiters,outshape,stride) + b

def conv2d(x,shape,stride):
    fiters = tf.get_variable('fiters',shape = shape,initializer= tf.random_normal_initializer(stddev = 0.02))
    b = tf.get_variable('b',shape = [shape[-1]],initializer= tf.constant_initializer ([0]))
    return tf.nn.conv2d(x,fiters,stride,'SAME') + b

def fconnect(x,out_num):
    w = tf.get_variable('w',[x.shape[-1],out_num],initializer=tf.random_normal_initializer (stddev = 0.02) )
    b = tf.get_variable('b',[out_num ],initializer= tf.constant_initializer([0]) )
    return tf.matmul(x,w) + b



def leak_relu(x, s):
    return tf.maximum(x,s*x)

def mapping(x):    #线性映射，将图像保持在0，255之间
    max = np.max(x)
    min = np.min(x)
    return (x - min) * 255.0 / (max - min + epclion)

def batch_norm(x):
    mu, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    beta = tf.get_variable("beta", [mu.shape[-1]], initializer=tf.constant_initializer(1.0))
    gamma = tf.get_variable("gamma", [mu.shape[-1]], initializer=tf.constant_initializer(0))
    return (x - mu) * beta / tf.sqrt(var + epclion) + gamma

class Generator:
    def __init__(self,name):
        self.name = name
    def __call__(self,x,is_training = True ):
        with tf.variable_scope(name_or_scope= self.name,reuse = False ):
            with tf.variable_scope(name_or_scope= "linear"):
                inputs = tf.reshape(tf.nn.relu(fconnect(x,4*4*512)),[batch_size ,4,4,512])
            with tf.variable_scope(name_or_scope= 'deconv1'):
                deconv = deconv2(inputs, [5, 5, 256, 512], [1, 2, 2, 1], [batch_size, 8, 8, 256])
                inputs = tf.nn.relu(batch_norm(deconv))
            with tf.variable_scope(name_or_scope= 'deconv2'):
                deconv = deconv2(inputs,[5,5,128,256],[1,2,2,1],[batch_size ,16,16,128])
                inputs = tf.nn.relu(batch_norm(deconv))
            with tf.variable_scope(name_or_scope='deconv3'):
                deconv = deconv2(inputs,[5,5,64,128],[1,2,2,1],[batch_size ,32,32,64])
                inputs = tf.nn.relu(batch_norm(deconv))
            if H == 32:
                with tf.variable_scope(name_or_scope='deconv4'):
                    inputs = tf.nn.tanh(deconv2(inputs, [5, 5, C, 64], [1, 1, 1, 1], [batch_size, H, W, C]))
                return inputs
            with tf.variable_scope(name_or_scope='deconv4'):
                deconv = deconv2(inputs,[5,5,C,64],[1,2,2,1],[batch_size ,H,W,C])
                inputs = tf.nn.tanh(deconv)
            if H == 64:
                return inputs

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Discriminator:
    def __init__(self,name):
        self.name = name
    def __call__(self,x,reuse,is_training = True ):
        with tf.variable_scope(name_or_scope= self.name,reuse = reuse):
            with tf.variable_scope(name_or_scope='conv1'):
                conv = conv2d(x,[5,5,C,64],[1,2,2,1])
                inputs = leak_relu(batch_norm(conv),0.2)
            with tf.variable_scope(name_or_scope= 'conv2'):
                conv = conv2d(inputs,[5,5,64,128],[1,2,2,1])
                inputs = leak_relu(batch_norm(conv),0.2)
            with tf.variable_scope(name_or_scope= 'conv3'):
                conv = conv2d(inputs,[5,5,128,256],[1,2,2,1])
                inputs = leak_relu(batch_norm(conv),0.2)
            with tf.variable_scope(name_or_scope= 'conv4'):
                conv = conv2d(inputs,[5,5,256,512],[1,2,2,1])
                inputs = leak_relu(batch_norm(conv),0.2)
            inputs = tf.layers.flatten(inputs)
            return fconnect(inputs,1)

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



class GAN:
    def __init__(self):
        self.z = tf.placeholder('float',shape = [batch_size ,100])
        self.imag = tf.placeholder('float',shape = [batch_size ,H,W,C])
        self.is_training = tf.placeholder(tf.bool)
        D = Discriminator('discriminator')
        G = Generator('generator')
        self.fake_imag = G(self.z, self.is_training)
        self.fake_logit = tf.nn.sigmoid(D(self.fake_imag,False, self.is_training))
        self.real_logit = tf.nn.sigmoid(D(self.imag,True, self.is_training))
        self.dloss = - (tf.reduce_mean(tf.log(self.real_logit + epclion)) + tf.reduce_mean(tf.log(1-self.fake_logit + epclion )))
        self.gloss = - tf.reduce_mean(tf.log(self.fake_logit + epclion))
        self.opg = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5).minimize(self.gloss ,var_list=G.var)
        self.opd = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5).minimize(self.dloss ,var_list=D.var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self):
        saver = tf.train.Saver()
        num = 200
        data = sio.loadmat('./facedata.mat')['data']
        for n in range(num):
            for i in range(int(len(data)//batch_size - 1 )):
                batch = data[i*batch_size : i*batch_size + batch_size,:,:,:] / 127.5 - 1.0
                z = np.random.standard_normal([batch_size ,100])
                dloss = self.sess.run(self.dloss,feed_dict = {self.imag:batch,self.z:z, self.is_training: False})
                gloss = self.sess.run(self.gloss,feed_dict = {self.imag:batch,self.z:z, self.is_training: False})
                self.sess.run(self.opd ,feed_dict= {self.imag:batch,self.z:z, self.is_training: True})
                self.sess.run(self.opg, feed_dict={self.imag: batch, self.z: z, self.is_training: True})
                if i % 10 == 0:
                    print('num:%d,step:%d,dloss:%g,gloss:%g'% (n,i,dloss,gloss))
                    z = np.random.standard_normal([batch_size ,100])
                    image = self.sess.run(self.fake_imag,feed_dict={self.imag: batch, self.z: z, self.is_training: False})
                    for j in range(batch_size ):
                        if C == 1:
                            Image.fromarray(np.reshape(np.uint8(mapping(image[j, :, :, :])), [H, W])).save("./result0//" + str(n) + "_" + str(j) + ".jpg")
                        else:
                            Image.fromarray(np.uint8(mapping(image[j, :, :, :]))).save("./result0//" + str(n) + "_" + str(j) + ".jpg") #数组转换为图像使用Image.fromarray
            saver.save(self.sess, "./para//model.ckpt")
if __name__ == "__main__":
    gan = GAN()
    gan()
