# univariate multi-step vector-output stacked lstm example
import numpy
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

numpy.random.seed(7) 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
#
def insert_end(Xin,new_input):
    for i in range(n_steps_in-1):
        Xin[:,i,:] = Xin[:,i+1,:]
    Xin[:,n_steps_in-1,:] = new_input
    return Xin
# define input sequence
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
raw_seq = read_csv('single_site_fof2_2001_1.csv', header=0, squeeze=True)
raw_seq=raw_seq[0:720*2]
in_seq=raw_seq.Values.values/10
in_seq=in_seq.reshape(len(in_seq),1)
scaler = MinMaxScaler(feature_range=(0,1))
in_seq = scaler.fit_transform(in_seq)
# choose a number of time steps
n_steps_in, n_steps_out = 24, 1
# split into samples
X, Y = split_sequence(in_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
Y=Y.reshape(Y.shape[0],n_steps_out)
train_percent=0.9
trainidx=int(train_percent*len(X))
Xtrain = X[:trainidx,:,:]  
Ytrain = Y[:trainidx,:]
Xtest = X[trainidx:,:,:]  
Ytest= Y[trainidx:,:]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, Y, epochs=100,batch_size=300, verbose=1)
# demonstrate prediction
#x_input = array([70, 80, 90])
#x_input = x_input.reshape((1, n_steps_in, n_features))
#yhat = model.predict(x_input, verbose=0)
#print(yhat)

predictions= model.predict(Xtrain, batch_size=1)
#plt.figure(figsize=(30,15))
pyplot.plot(predictions[:,0], '.-r', linewidth=1)
pyplot.plot(Ytrain[:,0], '.-b', linewidth=1)
pyplot.show()

model.reset_states()
preds = model.predict(Xtest,batch_size=1)
preds = scaler.inverse_transform(preds)

Ytest=numpy.asanyarray(Ytest)  
#Ytest=Ytest.reshape(-1,1) 
Ytest = scaler.inverse_transform(Ytest)
Ytrain=numpy.asanyarray(Ytrain)  
#Ytrain=Ytrain.reshape(-1,1) 
Ytrain = scaler.inverse_transform(Ytrain)

predictions=numpy.asanyarray(predictions)  
#predictions=predictions.reshape(-1,1) 
predictions = scaler.inverse_transform(predictions)
#
pyplot.plot(Ytest[:,0],'.-b')
pyplot.plot(preds[:,0], '.-r')
pyplot.show()

pyplot.plot(Ytest[100,:],'.-b')
pyplot.plot(preds[100,:], '.-r')
#
beg=0 #trainidx
forcasted_output = []
model.reset_states()

#Xin = Xtrain[beg:beg+1,:,:].copy()
Xin = Xtest[beg:beg+1,:,:].copy()
for i in range(360):
    out = model.predict(Xin, batch_size=1)    
    forcasted_output.append(out[0,:])
    Xin = insert_end(Xin,out[0,0])
model.reset_states()
forcasted_output=numpy.asanyarray(forcasted_output)   
#forcasted_output=forcasted_output.reshape(-1,1) 
forcasted_output = scaler.inverse_transform(forcasted_output)

#pyplot.plot(Ytrain , '.-b', linewidth=1)
#pyplot.plot(predictions , '.-g', linewidth=1)
pyplot.plot(Ytest[:,0] , '.-b', linewidth=1)
pyplot.plot(preds[:,0] , '.-g', linewidth=1)
pyplot.plot(forcasted_output[:,0],'.-r' , linewidth=1)
pyplot.legend(('test','Forcasted1','Forcasted2'))
pyplot.show()

pyplot.plot(Ytest[100,:] , '.-b', linewidth=1)
pyplot.plot(preds[100,:] , '.-g', linewidth=1)
pyplot.plot(forcasted_output[100,:],'.-r' , linewidth=1)
pyplot.legend(('test','Forcasted1','Forcasted2'))