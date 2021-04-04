import tensorflow as tf

y=1.
n=-1.
bias=1.0
x=[[y,y,bias],
    [y,n,bias],
    [n,y,bias],
    [n,n,bias]]
out=[[y],[y],[y],[n]]

w=tf.Variable(tf.random_normal([3,1]))

def step(x):
    is_greater=tf.greater(x,0)
    as_float=tf.to_float(is_greater)
    doubled=tf.multiply(as_float,2)
    return tf.subtract(doubled,1)
sess=tf.Session()
sess.run(tf.initialize_all_variables())
output=step(tf.matmul(x,w))
print(sess.run(output))
error=tf.subtract(out,output)
mse=tf.reduce_mean(tf.square(error))

delta=tf.matmul(x,error,transpose_a=True)
print(sess.run(delta))
train=tf.assign(w,tf.add(w,delta))
print(sess.run(train))
sess=tf.Session()
sess.run(tf.initialize_all_variables())

err,target = 1,0
epoch,max_epoch=0,10
while(err>target and epoch < max_epoch):
    epoch+=1
    err,_=sess.run([mse,train])
    print('epoch:', epoch, 'mse:',err,'value:',_)



    
    
       

