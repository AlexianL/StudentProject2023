import numpy as np 
import pandas as pd
import csv
#Libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set(color_codes = True)
sns.set(font_scale=1.5) #fixing font size

#Libraries for artificial neural network
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Normalization
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import Callback
from tqdm import tqdm

##################################################### RAW DATA

#Open AME mass table document
mass_file = open("1_raw_data/mass_1.mas20.txt","r+")


#We create the .csv file and give the name of the columns
mass_csv = open("2_processed_data/mass_data.csv","w+")
ame_csv_header_row = "N-Z;N;Z;A;ame_ME;ame_ME_unc;ame_BE/A;ame_BE/A_unc;ame_BDE;ame_BDE_unc;ame_AM;ame_AM_unc\n"
mass_csv.writelines(ame_csv_header_row)

#Extract data from AME mass data into a pandas dataframe and csv file
#We should have 3378 entries in .csv doc, so 3377 elements in it
#As the first line is the column names

element_list = mass_file.readlines()


#The following lines are for the purpose of standardization of the data
#As the data is in a complicated format (some values are empty) and thus
#we need to process the data ourselves

for element in element_list :

    
    splitted_line = element.split() #Split a string separated by spaces
    #We will get a list of 15 elements in the end

    #All elements have a column with only "B-" written
    if splitted_line.index("B-") == 11 : 
        #We want to get rid of indices 0 and 6
        splitted_line.pop(0) 
        splitted_line.pop(5) #5 as index 0 is already removed by .pop(0)
    
    if splitted_line.index("B-") == 10 :
        if (int(splitted_line[1]) - int(splitted_line[2]) == int(splitted_line[0]) and 
            int(splitted_line[1]) + int(splitted_line[2]) == int(splitted_line[3])) :
            splitted_line.pop(5)
        else : #The only other possibility is 9 which is what we look for
            splitted_line.pop(0)
    
    if len(splitted_line) != 15 :
        #Beta-decay energies uncertainties are sometimes empty, we add a 0
        splitted_line.insert(11,"0") 
        

    
    #We get rid of element symbol and "B-" in the list
    #We now have list of 13 elements
    if splitted_line[10].find("*") != -1 :
        splitted_line[10] = "0" #Replace "*" by "0"
    
    splitted_line.pop(4) #Getting rid of element symbols
    splitted_line.pop(8) #Getting rid of "B-" string

    #Values for atomic_mass follow a strange format
    #We thus concatenate two columns
    #index 10 & 11
    atomic_mass_coma = splitted_line.pop(11)
    atomic_mass_coma = "." + atomic_mass_coma.replace(".","")
    splitted_line[10] = splitted_line[10] + atomic_mass_coma
    
    
    #We now have list of 12 elements

    #Remove "#" and standardization of the list in order to convert into array
    for i in range(12) :
        if splitted_line[i].find("#") != -1 :
            splitted_line[i] = splitted_line[i].replace("#","")

    mass_csv.writelines(";".join(splitted_line) + "\n")
    
#Open DZ10 document
duzu_file=open("1_raw_data/duzu.txt","r+")

#Extract data from DZ10 into a .csv file

dz_element_list=duzu_file.readlines()

dz_csv=open("2_processed_data/dz_data.csv","w+")
dz_csv_header_row="Z;N;dz_BE/A;dz_ME\n"
dz_csv.writelines(dz_csv_header_row)

for element in dz_element_list :
    dz_split_line=element.split() 
    dz_split_line.pop(0)
    dz_split_line.pop(1)
    dz_split_line.pop(2)
    dz_split_line.pop(3)

    if not(dz_split_line[2].find("NaN")!=-1 or 
           dz_split_line[3].find("NaN")!=-1 or
           np.float128(dz_split_line[2])<0) :  #Negative binding energy

           dz_csv.writelines(";".join(dz_split_line)+"\n")


####################################################### DATA PROCESSING

dz_data = pd.read_csv("2_processed_data/dz_data.csv", sep=";")
#We create a new column in pandas dataframe "dz_data" containing mass number
dz_data["A"] = dz_data["Z"] + dz_data["N"]

#We create a new column containing Binding Energy not divided by A
dz_data["dz_BE"] = dz_data["dz_BE/A"] * dz_data["A"]

dz_data.sort_values(by=['Z'], ascending=True)
dz_data['dz_S1n'] = dz_data['dz_BE'] - dz_data['dz_BE'].shift(1)


dz_data = dz_data.sort_values(by=['N','A'], ascending=True)
dz_data['dz_S1p'] = dz_data['dz_BE'] - dz_data['dz_BE'].shift(1)

#Data already well sorted by the computation of S1p

dz_data['dz_S2p'] = dz_data['dz_BE'] - dz_data['dz_BE'].shift(2)

dz_data = dz_data.sort_values(by=['Z','A'], ascending=True)
dz_data['dz_S2n'] = dz_data['dz_BE'] - dz_data['dz_BE'].shift(2)


ame_data = pd.read_csv("2_processed_data/mass_data.csv", sep=";")

ame_data['ame_BE'] = ame_data['ame_BE/A'] * ame_data['A']

ame_data = ame_data.sort_values(by=['N','Z'], ascending=True)
ame_data['ame_S2p'] = ame_data['ame_BE'] - ame_data['ame_BE'].shift(2)
ame_data['ame_S1p'] = ame_data['ame_BE'] - ame_data['ame_BE'].shift(1)

ame_data = ame_data.sort_values(by=['Z','N'], ascending=True)
ame_data['ame_S2n'] = ame_data['ame_BE'] - ame_data['ame_BE'].shift(2)
ame_data['ame_S1n'] = ame_data['ame_BE'] - ame_data['ame_BE'].shift(1)

merged_data=pd.merge(dz_data, ame_data, on=["Z", "N", "A"])

#Change of units
#AME file gives energies in keV
#DZ10 file gives energies in MeV
#We want MeV at the end
merged_data["ame_BE/A"] = merged_data["ame_BE/A"]/1000
merged_data["ame_ME"] = merged_data["ame_ME"]/1000
merged_data["ame_BE"] = merged_data["ame_BE"]/1000
merged_data["ame_S2p"] = merged_data["ame_S2p"]/1000
merged_data["ame_S2n"] = merged_data["ame_S2n"]/1000
merged_data["ame_S1p"] = merged_data["ame_S1p"]/1000
merged_data["ame_S1n"] = merged_data["ame_S1n"]/1000


merged_data["Surf"] = np.power(merged_data["A"],2/3)
merged_data["Asym"] = ( (merged_data["N"]-merged_data["Z"])**2 ) / merged_data["A"]
merged_data["Coul"] = ( merged_data["Z"]*(merged_data["Z"]-1) ) / np.power(merged_data["A"],1/3)
merged_data["Pair"] = np.power(merged_data["A"],-1/2)
merged_data["Z_parity"] = np.power(-1,merged_data["Z"])
merged_data["N_parity"] = np.power(-1,merged_data["N"])

#We compute the difference in binding energy AME - DZ10
merged_data["BE_diff_dz_ame"] = merged_data["ame_BE/A"]*merged_data["A"] - merged_data["dz_BE"]

merged_data["Z_parity"] = np.power(-1,merged_data["Z"])
merged_data["N_parity"] = np.power(-1,merged_data["N"])

magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184]

#Adding two columns for the distance with respect to magic numbers
merged_data["Z_distance"] = None
merged_data["N_distance"] = None

#Compute the distance to magic numbers
for i, row in merged_data.iterrows():
    z = row["Z"]
    n = row["N"]
    merged_data.at[i, "Z_distance"] = min([abs(z - m) for m in magic_numbers])
    merged_data.at[i, "N_distance"] = min([abs(n - m) for m in magic_numbers])

#We save this merged dataframe to .csv

merged_data.to_csv("2_processed_data/merged_data.csv",sep=";", index=False)

merged_data.drop(merged_data[(merged_data["ame_S2n"]<0 )].index, inplace=True)
merged_data.drop(merged_data[(merged_data["ame_S2p"]<0 )].index, inplace=True)
merged_data.drop(merged_data[(merged_data["ame_S1n"]<0 )].index, inplace=True)
merged_data.drop(merged_data[(merged_data["ame_S1p"]<0 )].index, inplace=True)


#From the merged table, create one training dataset and a validation one
train_data = pd.DataFrame(columns=["Z","N","dz_BE/A","dz_ME","A","dz_BE","dz_S1n","dz_S1p","dz_S2p", "dz_S2n","ame_ME", "ame_BE/A", "ame_AM", "ame_BE", "ame_S2p", "ame_S2n", "BE_diff_dz_ame","Surf","Asym","Coul","Pair","Z_parity","N_parity","Z_distance","N_distance"])
test_data = pd.DataFrame(columns=["Z","N","dz_BE/A","dz_ME","A","dz_BE","dz_S1n","dz_S1p","dz_S2p", "dz_S2n","ame_ME", "ame_BE/A", "ame_AM", "ame_BE", "ame_S2p", "ame_S2n", "BE_diff_dz_ame","Surf","Asym","Coul","Pair","Z_parity","N_parity","Z_distance","N_distance"])


#We separate the merged dataframe into training and validation datasets
for i in range(len(merged_data)) :
    
    if int(merged_data.iloc[i]["Z"]) in [10,38,54,68,82] :
        test_data = test_data.append(merged_data.iloc[i], ignore_index=True)

    else :
        train_data = train_data.append(merged_data.iloc[i], ignore_index=True)


#We don't use data before A<16 because these light nuclei experience
#At least for training
#Physics effects that are very far from trivial (halo etc)
train_data.drop(train_data[(train_data["A"]<16 )].index, inplace=True)
train_data.drop(train_data[(train_data["ame_S2n"]<0 )].index, inplace=True)
train_data.drop(train_data[(train_data["ame_S2p"]<0 )].index, inplace=True)
test_data.drop(test_data[(test_data["ame_S2n"]<0 )].index, inplace=True)
test_data.drop(test_data[(test_data["ame_S2p"]<0 )].index, inplace=True)

train_data.drop(train_data[(train_data["ame_S1n"]<0 )].index, inplace=True)
train_data.drop(train_data[(train_data["ame_S1p"]<0 )].index, inplace=True)
test_data.drop(test_data[(test_data["ame_S1n"]<0 )].index, inplace=True)
test_data.drop(test_data[(test_data["ame_S1p"]<0 )].index, inplace=True)


train_data.to_csv("2_processed_data/train_merged_data.csv",sep=";")
test_data.to_csv("2_processed_data/validation_merged_data.csv",sep=";")

####################################################### DATA RESCALING

merged_data = pd.read_csv("2_processed_data/merged_data.csv", sep=";")

scaler = MinMaxScaler(feature_range=(0,1))

def rescale(list) :
    """This function adds new columns to the merged dataframe with data rescaled
    between 0 and 1"""
    for column in list :
        merged_data["rescaled_"+column]=scaler.fit_transform(pd.Series.to_numpy(merged_data[column]).reshape(-1,1))

columns=["ame_BE","N","Z","Surf","Asym","Coul","Pair","Z_parity","N_parity","Z_distance","N_distance", "ame_S1p", "ame_S1n", "ame_S2p", "ame_S2n"]

rescale(columns)

merged_data.to_csv("3_rescaled_data/rescaled_data.csv",sep=";", index=False)

merged_data.drop(merged_data[(merged_data["ame_S2n"]<0 )].index, inplace=True)
merged_data.drop(merged_data[(merged_data["ame_S2p"]<0 )].index, inplace=True)

merged_data.drop(merged_data[(merged_data["ame_S1n"]<0 )].index, inplace=True)
merged_data.drop(merged_data[(merged_data["ame_S1p"]<0 )].index, inplace=True)

#From the merged table, create one training dataset and a validation one
#Not sure the next two lines are useful
train_data = pd.DataFrame(columns=["Z","N","dz_BE/A","dz_ME","A","dz_BE","dz_S1n","dz_S1p","dz_S2p", "dz_S2n","ame_ME", "ame_BE/A", "ame_AM", "ame_BE", "ame_S1p", "ame_S1n", "ame_S2p", "ame_S2n", "BE_diff_dz_ame","Surf","Asym","Coul","Pair","Z_parity","N_parity","Z_distance","N_distance"])
test_data = pd.DataFrame(columns=["Z","N","dz_BE/A","dz_ME","A","dz_BE","dz_S1n","dz_S1p","dz_S2p", "dz_S2n","ame_ME", "ame_BE/A", "ame_AM", "ame_BE", "ame_S1p", "ame_S1n", "ame_S2p", "ame_S2n", "BE_diff_dz_ame","Surf","Asym","Coul","Pair","Z_parity","N_parity","Z_distance","N_distance"])


#We separate the merged dataframe into training and validation datasets
for i in range(len(merged_data)) :
    
    if int(merged_data.iloc[i]["Z"]) in [10,38,54,68,82] :
        test_data = test_data.append(merged_data.iloc[i], ignore_index=True)

    else :
        train_data = train_data.append(merged_data.iloc[i], ignore_index=True)


#We don't use data before A<16 because these light nuclei experience
#At least for training
#Physics effects that are very far from trivial (halo etc)
train_data.drop(train_data[(train_data["A"]<16 )].index, inplace=True)
train_data.drop(train_data[(train_data["ame_S2n"]<0 )].index, inplace=True)
train_data.drop(train_data[(train_data["ame_S2p"]<0 )].index, inplace=True)
test_data.drop(test_data[(test_data["ame_S2n"]<0 )].index, inplace=True)
test_data.drop(test_data[(test_data["ame_S2p"]<0 )].index, inplace=True)

train_data.drop(train_data[(train_data["ame_S1n"]<0 )].index, inplace=True)
train_data.drop(train_data[(train_data["ame_S1p"]<0 )].index, inplace=True)
test_data.drop(test_data[(test_data["ame_S1n"]<0 )].index, inplace=True)
test_data.drop(test_data[(test_data["ame_S1p"]<0 )].index, inplace=True)


train_merged_csv = train_data.to_csv("3_rescaled_data/train_rescaled_data.csv",sep=";")
validation_merged_csv = test_data.to_csv("3_rescaled_data/validation_rescaled_data.csv",sep=";")

####################################################### ARTIFICIAL NEURAL NETWORK

train_data = pd.read_csv("3_rescaled_data/train_rescaled_data.csv", sep=";")

#First inputs
target = train_data["rescaled_ame_BE"]
n_input = train_data["rescaled_N"]
z_input = train_data["rescaled_Z"]

#Liquid drop inputs
surf_input = train_data["rescaled_Surf"]
asym_input = train_data["rescaled_Asym"]
coul_input = train_data["rescaled_Coul"]
pair_input = train_data["rescaled_Pair"]

#Other inputs that may help
z_parity_input = train_data["rescaled_Z_parity"]
n_parity_input = train_data["rescaled_N_parity"]
z_distance_input = train_data["rescaled_Z_distance"]  
n_distance_input = train_data["rescaled_N_distance"]
S1p_input = train_data["rescaled_ame_S1p"]
S1n_input = train_data["rescaled_ame_S1n"]
S2p_input = train_data["rescaled_ame_S2p"]
S2n_input = train_data["rescaled_ame_S2n"]

#Creation of a fonction that create an ANN

def create_model(num_inputs, num_layers, num_neurons):
    inputs = [keras.layers.Input(shape=(1,)) for i in range(num_inputs)]
    merged = keras.layers.Concatenate()(inputs)

    dense = merged
    for i in range(num_layers):
        dense = Dense(num_neurons, activation="relu")(dense)
    
    output = Dense(1, activation="relu")(dense)
    model = keras.models.Model(inputs, output)
    return model


model4 = create_model(12,14,100) ################## model can be modified here

model4.compile(optimizer=Adam(learning_rate=0.00001), loss="mean_squared_error")

#Class which will be usefull to keep the best trained model

class EarlyStoppingByLossValue(Callback):
    def __init__(self, value=0.00000009):
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get("loss")
        if current_loss < self.value:
            self.model.stop_training = True
            print("Early stopping by loss value at epoch", epoch)

early_stopping = EarlyStoppingByLossValue()

history4=model4.fit(x=([n_input, z_input, surf_input, coul_input, asym_input, pair_input, S1n_input, S1p_input, S2n_input, S2p_input, z_distance_input, n_distance_input]), y=target, epochs=5000, shuffle=True, verbose=2, callbacks=[early_stopping] )

scaler = MinMaxScaler(feature_range=(0,1))

#rescaled_data = pd.read_csv("/3_rescaled_data/rescaled_data.csv"), sep=";")
rescaled_data = pd.read_csv("3_rescaled_data/rescaled_data.csv", sep=";")

rescaled_target = scaler.fit_transform(pd.Series.to_numpy(rescaled_data["ame_BE"]).reshape(-1,1))


train_predictions = model4.predict(x=([n_input, z_input, surf_input, coul_input, asym_input, pair_input, S1n_input, S1p_input,  S2n_input, S2p_input, z_distance_input, n_distance_input]), verbose=0)


train_rescaled_predictions = [(i - scaler.min_)/scaler.scale_ for i in train_predictions]


train_data["BE_Predictions"] = np.double(train_rescaled_predictions)
train_data["Difference_BE_AME_ANN"] = train_data["ame_BE"] - train_data["BE_Predictions"]
train_data["Difference_BE_DZ_AME"] = train_data["dz_BE"] - train_data["ame_BE"]
train_data["Difference_BE_DZ_ANN"] = train_data["dz_BE"] - train_data["BE_Predictions"]

test_data = pd.read_csv("3_rescaled_data/validation_rescaled_data.csv", sep=";")


test_target = test_data["rescaled_ame_BE"]
test_n_input = test_data["rescaled_N"]
test_z_input = test_data["rescaled_Z"]
test_coul_input = test_data["rescaled_Coul"]
test_surf_input = test_data["rescaled_Surf"]
test_asym_input = test_data["rescaled_Asym"]
test_pair_input = test_data["rescaled_Pair"]
test_z_parity_input = test_data["rescaled_Z_parity"]
test_n_parity_input = test_data["rescaled_N_parity"]
test_z_distance_input = test_data["rescaled_Z_distance"]
test_n_distance_input = test_data["rescaled_N_distance"]
test_S1p_input = test_data["rescaled_ame_S1p"]
test_S1n_input = test_data["rescaled_ame_S1n"]
test_S2p_input = test_data["rescaled_ame_S2p"]
test_S2n_input = test_data["rescaled_ame_S2n"]

validation_predictions = model4.predict(x=([test_n_input, test_z_input, test_surf_input, test_coul_input, test_asym_input, test_pair_input,  test_S1n_input, test_S1p_input, test_S2n_input, test_S2p_input, test_z_distance_input, test_n_distance_input]))

validation_rescaled_predictions = [ (i - scaler.min_)/scaler.scale_ for i in validation_predictions]

test_data["BE_Predictions"] = np.double(validation_rescaled_predictions)
test_data["Difference_BE_AME_ANN"] = test_data["ame_BE"] - test_data["BE_Predictions"]
test_data["Difference_BE_DZ_ANN"] = test_data["dz_BE"] - test_data["BE_Predictions"]

#Sp,Sn,S2p,S2n predicted

train_data=train_data.sort_values(by=['A','N'], ascending=True)
train_data.head(15)

train_data=train_data.sort_values(by=['N','Z'], ascending=True)
train_data['Prediction_S2p'] = train_data['BE_Predictions'] - train_data['BE_Predictions'].shift(2)

train_data = train_data.sort_values(by=['A','N'], ascending=True)
train_data['Prediction_S2n'] = train_data['BE_Predictions'] - train_data['BE_Predictions'].shift(2)
train_data['Prediction_S1n'] = train_data['BE_Predictions'] - train_data['BE_Predictions'].shift(1)

train_data["Difference_S2n_AME_Predictions"] = train_data["ame_S2n"] - train_data["Prediction_S2n"]
train_data["Difference_S2p_AME_Predictions"] = train_data["ame_S2p"] - train_data["Prediction_S2p"]

train_data["Difference_S2n_DZ_Predictions"] = train_data["dz_S2n"] - train_data["Prediction_S2n"]
train_data["Difference_S2p_DZ_Predictions"] = train_data["dz_S2p"] - train_data["Prediction_S2p"]

test_data.sort_values(by=['N','Z'], ascending=True)
test_data['Prediction_S2p'] = test_data['BE_Predictions'] - test_data['BE_Predictions'].shift(2)

test_data = test_data.sort_values(by=['A','N'], ascending=True)
test_data['Prediction_S2n'] = test_data['BE_Predictions'] - test_data['BE_Predictions'].shift(2)
test_data['Prediction_S1n'] = test_data['BE_Predictions'] - test_data['BE_Predictions'].shift(1)

test_data["Difference_S2n_AME_Predictions"] = test_data["ame_S2n"] - test_data["Prediction_S2n"]
test_data["Difference_S2p_AME_Predictions"] = test_data["ame_S2p"] - test_data["Prediction_S2p"]

test_data["Difference_S2n_DZ_Predictions"] = test_data["dz_S2n"] - test_data["Prediction_S2n"]
test_data["Difference_S2p_DZ_Predictions"] = test_data["dz_S2p"] - test_data["Prediction_S2p"]

train_data.to_csv("4_final_data/train_final_data.csv",sep=";")
test_data.to_csv("4_final_data/validation_final_data.csv",sep=";")

##################################################### PLOT

plt.rcParams['text.usetex'] = True

final_train = pd.read_csv("4_final_data/train_final_data.csv", sep=";")
final_test = pd.read_csv("4_final_data/validation_final_data.csv", sep=";")
merge_data = pd.read_csv("2_processed_data/merged_data.csv", sep=";")

merge_data.drop(merge_data[(merge_data["A"]<16 )].index, inplace=True)
merge_data["Difference_BE_DZ_AME"] = merge_data["dz_BE"] - merge_data["ame_BE"]

plt.figure(figsize =(20,10))
plt.ylabel("BE(EXP) - BE(DZ10) (MeV)")

sns.scatterplot(x="A",y="Difference_BE_DZ_AME",data=merge_data, palette="rainbow_r", label="EXP - DZ10")
plt.savefig("5_plots/diff_BE(exp)-BE(DZ).png")

## Train plot

rms_train = np.sqrt(((final_train["Difference_BE_AME_ANN"] ** 2).sum()) / len(final_train["Difference_BE_AME_ANN"]))

print('RMS AME - Predict:', rms_train)

plt.figure(figsize =(16,8))
plt.ylabel('$\Delta$ BE (MeV)')
plt.text(200, -4,'RMS Exp - Predict (train): {:.4f}'.format(rms_train) )

sns.scatterplot(x="A",y="Difference_BE_DZ_AME",data=merge_data, palette="rainbow_r", label="EXP - DZ10")
sns.scatterplot(x="A",y="Difference_BE_AME_ANN",data=final_train, palette="rainbow_r", label="Exp - Predict")
plt.savefig("5_plots/diff_train.png")


## S2n plots for isotopic chains

final_test.drop(final_test[(final_test["Prediction_S2n"]<0 )].index, inplace=True)
final_test.drop(final_test[(final_test["Prediction_S2p"]<0 )].index, inplace=True)
final_test.drop(final_test[(final_test["Prediction_S2n"]>50 )].index, inplace=True)

def plot_S2n(data, Z_values):

    for Z in Z_values:
        plt.figure(figsize=(16, 8))
        plt.title(" S2n for Z={}".format(Z))
        plt.ylabel('$S_{2n}$ (MeV))')
        plt.legend()
        sns.lineplot(x="N", y="ame_S2n", data=data[data['Z'] == Z], color="black", label='EXP')
        sns.lineplot(x="N", y="Prediction_S2n", data=data[data['Z'] == Z], color="orange", label='ANN')
        sns.lineplot(x="N", y="dz_S2n", data=data[data['Z'] == Z], color="red", label='DZ10')
        plt.savefig("5_plots/S2n_Z_{}.png".format(Z))
        plt.show()
    

plot_S2n(final_test, [10, 38, 54, 68, 82])

# S1n plots

final_test.drop(final_test[(final_test["Prediction_S1n"]<0 )].index, inplace=True)
final_test.drop(final_test[(final_test["Prediction_S1n"]>50 )].index, inplace=True)


def plot_S1n(data, Z_values):

    for Z in Z_values:
        plt.figure(figsize=(16, 8))
        plt.title(" Sn for Z={}".format(Z))
        plt.ylabel('$S_n$ (MeV)')
        plt.legend()
        sns.lineplot(x="N", y="ame_S1n", data=data[data['Z'] == Z], color="black", label='EXP')
        sns.lineplot(x="N", y="Prediction_S1n", data=data[data['Z'] == Z], color="orange", label='ANN')
        sns.lineplot(x="N", y="dz_S1n", data=data[data['Z'] == Z], color="red", label='DZ10')
        plt.savefig("5_plots/S1n_Z_{}.png".format(Z))
        plt.show()
       
plot_S1n(final_test, [10, 38, 54, 68, 82])