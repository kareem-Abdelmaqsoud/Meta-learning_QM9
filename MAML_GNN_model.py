
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.optimizers as optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of collections of numpy arrays 
import tarfile
import urllib.request
import os.path
import matplotlib.pyplot as plt
import warnings
from keras.optimizers import SGD
import pickle
import random
import time
warnings.filterwarnings('ignore')


# In[3]:


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float(x):
    try:
        return float(x)
    except:
        return 0


def data_parse(record):
    features = {
        'N': tf.io.FixedLenFeature([], tf.int64),
        'labels': tf.io.FixedLenFeature([16], tf.float32),
        'elements': tf.io.VarLenFeature(tf.int64),
        'coords': tf.io.VarLenFeature(tf.float32),
    }
    parsed_features = tf.io.parse_single_example(
        serialized=record, features=features)
    coords = tf.reshape(tf.sparse.to_dense(
        parsed_features['coords'], default_value=0), [-1, 4])
    elements = tf.sparse.to_dense(parsed_features['elements'], default_value=0)
    return (elements, coords), parsed_features['labels']


def prepare_qm9_record(lines):
    pt = {'C': 6, 'H': 1, 'O': 8, 'N': 7, 'F': 9}
    N = int(lines[0])
    labels = [float(x) for x in lines[1].split('gdb')[1].split()]
    coords = np.empty((N, 4), dtype=np.float64)
    elements = [pt[x.split()[0]] for x in lines[2:N+2]]
    for i in range(N):
        coords[i] = [_float(x) for x in lines[i + 2].split()[1:]]
    feature = {
        'N': tf.train.Feature(int64_list=tf.train.Int64List(value=[N])),
        'labels': tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
        'elements': tf.train.Feature(int64_list=tf.train.Int64List(value=elements)),
        'coords': tf.train.Feature(float_list=tf.train.FloatList(value=coords.flatten())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def fetch_qm9():

    raw_filepath = 'qm9.tar.bz2'
    record_file = 'qm9.tfrecords'

    if os.path.isfile(record_file):
        print('Found existing record file, delete if you want to re-fetch')
        return record_file

    if not os.path.isfile(raw_filepath):
        print('Downloading qm9 data...', end='')
        urllib.request.urlretrieve(
            'https://ndownloader.figshare.com/files/3195389', raw_filepath)
        print('File downloaded')

    else:
        print(
            f'Found downloaded file {raw_filepath}, delete if you want to redownload')
    tar = tarfile.open(raw_filepath, 'r:bz2')

    print('')
    with tf.io.TFRecordWriter(record_file, options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        for i in range(1, 133886):
            if i % 100 == 0:
                print('\r {:.2%}'.format(i / 133886), end='')
            with tar.extractfile(f'dsgdb9nsd_{i:06d}.xyz') as f:
                lines = [l.decode('UTF-8') for l in f.readlines()]
                try:
                    writer.write(prepare_qm9_record(
                        lines).SerializeToString())
                except ValueError as e:
                    print(i)
                    raise e
    print('')
    return record_file


def get_qm9(record_file):
    return tf.data.TFRecordDataset(
        record_file, compression_type='GZIP').map(data_parse)


# In[4]:


np.random.seed(0)
qm9_records = fetch_qm9()
data = get_qm9(qm9_records)


# In[5]:


# convert the tfrecords to a usable format for this assignment
def convert_record(d):
    # break up record
    (ele, xyzc), y = d
    ele = ele.numpy()
    xyzc = xyzc.numpy()
    xyz = xyzc[:, :3]    
    # use nearest power of 2 (16)
    ohc = np.zeros((len(ele), 16))
    ohc[np.arange(len(ele)), ele - 1] = 1    
    return (ohc, xyz), y.numpy()


# In[6]:


# convert coordinates to pairwise distances
def x2e(x):
    '''convert xyz coordinates to inverse pairwise distance'''    
    r2 = jnp.sum((x - x[:, jnp.newaxis, :])**2, axis=-1)
    e = jnp.where(r2 != 0, 1 / r2, 0.)
    return e


# In[7]:


graph_feature_len = 8
node_feature_len = 16
msg_feature_len = 16

# make our weights
def init_weights(g, n, m):
    np.random.seed(1)
    we = np.random.normal(size=(n, m), scale=1e-1)
    np.random.seed(1)
    wv = np.random.normal(size=(m, n), scale=1e-1)
    np.random.seed(1)
    wu = np.random.normal(size=(n, g), scale=1e-1)
    return we, wv, wu

we, wv, wu = init_weights(graph_feature_len, node_feature_len, msg_feature_len)
np.random.seed(1)
features0 = np.random.normal(graph_feature_len)


# input --> molecular graph
# output --> fixed length vector 
def gnn(nodes, edges, we, wv, wu, features=features0):
    # step 1 --> compute features stored per edge based on current edge feature and node feature
    ek = jax.nn.relu(
        jnp.repeat(nodes[jnp.newaxis,...], nodes.shape[0], axis=0) @ we * edges[...,jnp.newaxis])
    # step 2 --> aggregate to node
    ebar = jnp.mean(ek, axis=1)
    # step 3 --> update node features
    v = jax.nn.relu(ebar @ wv) + nodes
    # step 4 --> update global feature
    global_node_features = jnp.sum(v, axis=0)
    u = jax.nn.relu(global_node_features  @ wu) + features
    return u

# euclidean distance between the vectors
def euclidean(x1, x2): # x1, x2 should be the same length, take a mean to maintain same magnitude
    return jnp.sqrt(jnp.mean((x1 - x2)**2))

def kernel_learning_model(x, train_x, w, b):
    # make vectorized version of kernel
    vkernel = jax.vmap(euclidean, in_axes=(None, 0), out_axes=0)
    # compute kernel with all training data
    s = vkernel(x, train_x)
    # dual form
    yhat = jnp.dot(s,w) + b
    return yhat

# make batched version that can handle multiple xs
batch_model = jax.vmap(kernel_learning_model, in_axes=(0, None, None, None), out_axes=0)


# In[8]:


tf.random.set_seed(1234)
data=data.shuffle(100000)
data = data.take(130)


# In[9]:


def feature_label(we, wv, wu, data):
    features = []
    y = []
    for d in data:
        (atom_ohc, coords), labels = convert_record(d)
        nodes = atom_ohc
        edges = x2e(coords)
        Features = gnn(nodes,edges, we, wv, wu)
        features.append(Features)
        y.append(labels)
    y = np.array(y)
    train_yms=[]
    train_yss= []
    for i in range(0,15):
        train_ym = np.mean(y[:,i])
        train_ys = np.std(y[:,i])
        y[:,i] = (y[:,i] - train_ym) / train_ys
        train_yms.append(train_ym)
        train_yss.append(train_ys)
    return features, np.array(y),np.array(train_yms),np.array(train_yss)


# In[10]:


def Feature(we, wv, wu, data):
    features = []
    for d in data:
        (atom_ohc, coords), labels = convert_record(d)
        nodes = atom_ohc
        edges = x2e(coords)
        Features = gnn(nodes,edges, we, wv, wu)
        features.append(Features)
    return features


# In[11]:


features,y,train_ym,train_ys = feature_label(we, wv, wu,data)


# In[12]:


def transform_prediction(y):
    return train_ys[1:][...,jnp.newaxis]*y  + train_ym[1:][...,jnp.newaxis]


# In[13]:


@jax.jit
def loss_train(w, b, train_x, x_1, y_1):
    return jnp.mean(jnp.abs(batch_model(x_1, train_x, w, b) - y_1))


# In[14]:


@jax.jit
def loss_val_test(w, b, train_x, x_2, y_2):
    return jnp.mean(jnp.abs(transform_prediction(batch_model(x_2, train_x, w, b)) - transform_prediction(y_2)))


# In[15]:


def inner_update(w, b, train_x, x_1, y_1,alpha):
    inner_grads= loss_grad_1(w, b, train_x, x_1, y_1)
    w -= alpha*inner_grads[0]/jnp.linalg.norm(inner_grads[0])
    b -= alpha*inner_grads[1]/jnp.linalg.norm(inner_grads[1])
    return w,b


# In[16]:


@jax.jit
def maml_loss(w,b,we,wv,wu,x_1,y_1, x_2, y_2,train_x,alpha, npoints,sample_idx):
    w,b = inner_update(w,b, train_x, x_1, y_1,alpha)
    features = Feature(we,wv,wu, data)
    train_x = jnp.array(features[:int(N * 0.9)])
    x_1 = train_x[sample_idx, :]
    x_2 = jnp.array(features[int(N * 0.9):])
    return loss_train(w, b, train_x, x_2, y_2)


# In[17]:


batch = jax.vmap(maml_loss, in_axes=(None, None, None, None,None,0,0,0,0, None,None,None, None), out_axes=0)


# In[18]:


def batch_maml_loss(w,b,we,wv,wu,x_1,y_1, x_2, y_2,train_x,alpha, npoints,sample_idx):
    task_losses= batch(w,b,we,wv,wu,x_1,y_1, x_2, y_2,train_x,alpha, npoints,sample_idx)
    return jnp.mean(task_losses)


# In[19]:


@jax.jit
def step(i,w,b,we,wv,wu,x_1,y_1, x_2, y_2,train_x,alpha, beta,npoints,sample_idx):
    g = loss_grad_2(w,b,we,wv,wu,x_1,y_1, x_2, y_2,train_x,alpha, npoints,sample_idx)   
    we -= g[0]*beta/jnp.linalg.norm(g[0])
    wv -= g[1]*beta/jnp.linalg.norm(g[1])
    wu -= g[2]*beta/jnp.linalg.norm(g[2])
    l = batch_maml_loss(w,b,we,wv,wu,x_1,y_1, x_2, y_2,train_x,alpha, npoints, sample_idx)
    return l,w,b,we,wv,wu


# In[20]:


loss_grad_1 = jax.grad(loss_train, (0,1))
loss_grad_2 = jax.grad(batch_maml_loss, (2,3,4))
we, wv, wu = init_weights(graph_feature_len, node_feature_len, msg_feature_len)
#hyperparameter for the inner loop (inner gradient update)
alpha= 0.00001
#hyperparameter for the outer loop (outer gradient update) i.e meta optimization
beta = 0.000001
num_tasks = 16


# In[21]:


def batching(y,train_x,npoints):
    xb_1 = []
    xb_2 = []
    yb_1 = []
    yb_2 = []
    negs= []
    for task in range(1,15):
        np.random.seed(1)
        sample_idx = np.random.choice(np.arange(train_x.shape[0]), replace=True, size=5*npoints)
        xb_1.append(train_x[sample_idx, :])
        train_y = np.array(y[:,task][:int(N * 0.9)])
        np.random.seed(1)
        neg = (1 if random.random() < 0.5 else -1)
        negs.append(neg)
        yb_1.append(neg*train_y[sample_idx])
        yb_2.append(neg*y[:,task][int(N * 0.9):])
        xb_2.append(np.array(features[int(N * 0.9):]))
        
    xb_1 = jnp.stack(xb_1)
    xb_2 = jnp.stack(xb_2)
    yb_1 = jnp.stack(yb_1)
    yb_2 = jnp.stack(yb_2)
    return xb_1,xb_2,yb_1,yb_2,negs


# In[22]:


N = y.shape[0]
train_x = np.array(features[:int(N * 0.9)])
x_2 = np.array(features[int(N * 0.9):])
y_2 = y[:,0][int(N * 0.9):]
def maml_train(npoints, task_idx,y):
    y = np.delete(y, task_idx, axis=1)
    np.random.seed(1)
    sample_idx = np.random.choice(np.arange(train_x.shape[0]), replace=True, size=5*npoints)
    x_1 = train_x[sample_idx, :]
    train_y = np.array(y[:,0][:int(N * 0.9)])
    y_1 = train_y[sample_idx]
    y_2 = y[:,0][int(N * 0.9):]
    np.random.seed(1)
    w = np.random.normal(size = train_y.shape)
    b = np.mean(train_y) 
    we, wv, wu = init_weights(graph_feature_len, node_feature_len, msg_feature_len)
    # batching tasks:
    xbs_1,xbs_2,ybs_1,ybs_2,negs = batching(y,train_x,npoints)
    batch_size= 3
    batch_num = 14//batch_size
    np.random.seed(1)
    sample_idx = np.random.choice(np.arange(xbs_1.shape[0]), replace=False, size=12)
    xbs_1 = xbs_1[sample_idx, :]
    ybs_1 = ybs_1[sample_idx]
    xbs_2 = xbs_2[sample_idx, :]
    ybs_2 = ybs_2[sample_idx]
    xbs_1 = xbs_1.reshape(batch_size,-1, xbs_1.shape[1], xbs_1.shape[-1])
    ybs_1 = ybs_1.reshape(batch_size,-1, ybs_1.shape[-1])
    xbs_2 = xbs_2.reshape(batch_size,-1, xbs_2.shape[1], xbs_2.shape[-1])
    ybs_2 = ybs_2.reshape(batch_size,-1, ybs_2.shape[-1])
    for i in range(100000):
        np.random.seed(1)
        for j in np.random.random_integers(0, batch_num - 1, size=batch_num):
            xb_1 = xbs_1[j]
            yb_1 = ybs_1[j]
            xb_2 = xbs_2[j]
            yb_2 = ybs_2[j]
            l,w,b,we,wv,wu = step(i,w,b,we,wv,wu,xb_1,yb_1, xb_2, yb_2,train_x,alpha, beta, npoints,sample_idx)
        if i%1000==0:
            print(l)
            print("------")
    print("task_done!")
    return w,b,we,wv,wu,negs


# In[ ]:


start_time = time.time()
num_samples= 20
optimal_param= []
#flag=True
for task_idx in range(0,15):
    optimal_param.append(maml_train(num_samples, task_idx,y))
#     _,_,we,wv,wu,_ = maml_train(num_samples, task_idx,y)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


optimal_param


# In[ ]:


import pickle
pickle.dump(optimal_param, open( "optimal_param.p", "wb" ))


# In[ ]:


optimal_param = pickle.load(open( "optimal_param.p", "rb" ))


# In[ ]:


eta= 0.0001


# In[ ]:


N = y.shape[0]
def maml_test(npoints, task_idx,seed):
    we,wv,wu,negs =optimal_param[task_idx]
    features = Feature(we,wv,wu, data)
    x_2 = np.array(features[int(N * 0.9):])
    y_2 = y[:,task_idx][int(N * 0.9):]
    train_x = np.array(features[:int(N * 0.9)])
    np.random.seed(seed)
    sample_idx = np.random.choice(np.arange(train_x.shape[0]), replace=True, size=5*npoints)
    x_1 = train_x[sample_idx, :]
    train_y = np.array(y[:,task_idx][:int(N * 0.9)])
    y_1 = train_y[sample_idx]
    np.random.seed(seed+200)
    w = np.random.normal(size = train_y.shape)
    b = np.mean(train_y) 
    for i in range(100000):
        inner_grads = loss_grad_1(w, b, train_x, x_1, y_1)
        for g in inner_grads:
            if jnp.linalg.norm(g)>= 1:
                g = 1*g/jnp.linalg.norm(g)
            else: 
                g = g
        w -= eta*inner_grads[0]
        b -= eta*inner_grads[1]
        if i%1000==0:
            print(loss_train(w, b, train_x, x_1, y_1))
    return loss_val_test(w, b, train_x, x_2, y_2)


# In[ ]:


start_time = time.time()
nvalues =[3, 5,10,20,50,100]
MAML_model_loss= []
np.random.seed(1)
seeds=np.random.randint(1,1000,15)
for task_idx in range(0,15):
    MAML_model_loss.append([maml_test(n,task_idx,seeds[task_idx]) for n in nvalues])
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


MAML_model_loss


# In[ ]:


MAML_model_loss


# In[ ]:


def transform_prediction(y):
    return train_ys[...,jnp.newaxis]*y  + train_ym[...,jnp.newaxis]


# In[ ]:


MAML_model_loss_actual = transform_prediction(MAML_model_loss)


# In[ ]:


pickle.dump(MAML_model_loss_actual, open( "MAML_model_loss_actual.p", "wb" ) )


# In[ ]:


MAML_model_loss_actual = pickle.load(open( "MAML_model_loss_actual_old.p", "rb" ) )


# In[ ]:


MAML_model_loss_actual.shape


# In[ ]:


MAML_model_loss_actual[14]


# In[ ]:


kernel_loss = pickle.load(open( "kernel_losses.p", "rb" ) )


# In[ ]:


tasks= ["Rotational constant A","Rotational constant B","Rotational constant C","Dipole moment(Debye)","Isotropic polarizability(Bohr^3)","HOMO(Hartree)","LUMO(Hartree)","Gap(Hartree)","Electronic spatial extent (Bohr^2)","Zero point vibrational energy(Hartree)","Internal energy at 0 K(Hartree)","Internal energy at 298.15 K(Hartree)","Enthalpy at 298.15 K(Hartree)","Free energy at 298.15 K(Hartree)","Heat capacity at 298.15 K(cal/mol.K)"]


# In[ ]:


def transform_prediction(y):
    return train_ys[...,jnp.newaxis]*y  + train_ym[...,jnp.newaxis]


# In[ ]:


kernel_loss_actual = transform_prediction(kernel_loss)


# In[ ]:


schnet_loss=np.array([0,0,0,0.033, 0.235, 0.001507365,0.00125001,0.002316195, 0.073,6.250050000000001e-05,0.00051471,0.000698535,0.00051471,0.00051471,0.033])
schnet_loss = np.reshape(jnp.repeat(schnet_loss,5),(15,5))


# In[ ]:


# kernel_loss_actual


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(14, 12), dpi=120)
nvalues =[3, 5,10,20,50]
for i in range(1, 16):
    plt.subplot(5, 3, i)
    plt.plot(nvalues,MAML_model_loss_actual[i-1], label= "MAML")
    plt.plot(nvalues,kernel_loss[i][:5], label = "Kernel")
    #plt.plot(nvalues,schnet_loss[i], label = "Schnet")
    plt.title(tasks[i-1])
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.tight_layout()
    plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(14, 12), dpi=120)
nvalues =[3, 5,10,20,50]
for i in range(1, 16):
    plt.subplot(5, 3, i)
    plt.plot(nvalues,MAML_model_loss_actual[i-1], label= "MAML")
    plt.plot(nvalues,kernel_loss[i][:5], label = "Kernel")
    plt.plot(nvalues,schnet_loss[i-1], label = "Schnet")
    plt.title(tasks[i-1])
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.tight_layout()
    plt.legend()
plt.show()


