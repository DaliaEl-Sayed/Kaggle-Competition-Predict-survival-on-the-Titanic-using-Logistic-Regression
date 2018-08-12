import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as op


def cost(initial_theta, x, y):

    z = x.dot(initial_theta).T
    sigmoid_z = 1.0 / (1.0 + np.exp(-1.0 * z))
    m_len = len(y)
    j = (1.0/m_len)*sum(-y.T.dot(np.log(sigmoid_z.T))-(1-y).T.dot(np.log(1-sigmoid_z.T)))
    return j


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#print(train_data.head())
#print(train_data.info())
#print(test_data.info())

################################### handle cabin NaN values ################################################

# cabine in both test and train data has too many NaN values with respect to known value so i will drop it
train_data = train_data.drop(['Cabin'], axis=1)
test_data = test_data.drop(['Cabin'], axis=1)

#print(train_data.head())
#print(train_data.info())
#print(test_data.info())

########################### time to fill nan value in Embarked and study it ########################################

Embarked_Nan_rows = train_data[train_data['Embarked'].isnull()] #same pclass and same ticket no so it is possiple that they came from same Embarked
data_emberked_of_Pclass_equal_one = train_data[(train_data['Pclass']==1)].filter(items=["Embarked"])
#print(data_emberked_of_Pclass_equal_one.groupby("Embarked").size()) #number of pclass = 1 and embarked = S is higher than C and Q so i will fill nan values with S
char_s = 'S'
train_data.loc[train_data["Embarked"].isnull(), 'Embarked'] = char_s
#check if there is any nan values left
#print(test_data.isnull().sum()) ##get all nan numbers for each col
#print(train_data.isnull().sum()) ##get all nan numbers for each col

#time to drow the mean of survived vs emberked for each class S C Q
Embarked = train_data[['Embarked', 'Survived']].groupby(["Embarked"], as_index= False).mean()
sns.barplot(x='Embarked', y='Survived', data=Embarked, order=['C', 'Q', 'S'], palette= "Set3")
plt.savefig("Embarked_Survival_plot.png") # Embarked c has the biggest survival chance

#let's convert Embarked class into numeric int num
train_data.loc[train_data["Embarked"]== 'C', 'Embarked'] = 0
test_data.loc[test_data["Embarked"]== 'C', 'Embarked'] = 0
train_data.loc[train_data["Embarked"]== 'Q', 'Embarked'] = 1
test_data.loc[test_data["Embarked"]== 'Q', 'Embarked'] = 1
train_data.loc[train_data["Embarked"]== 'S', 'Embarked'] = 2
test_data.loc[test_data["Embarked"]== 'S', 'Embarked'] = 2
#print(train_data.head())

################# time to handle fare missing data #################################################

#print(test_data.isnull().sum()) ##get all nan numbers for each col
#print(train_data.isnull().sum()) ##get all nan numbers for each col
Fare_Nan_rows = test_data[test_data['Fare'].isnull()] #get the NaN Fare data
#print(Fare_Nan_rows) # the NaN Fare has Pclass = 3 and Embarked = 2 so i will fill the missing one with median of Fare having pclass = 3 and Embarked = 2
#first combine train and test data
df1 = pd.DataFrame(train_data)
df2 = pd.DataFrame(test_data)
data = pd.concat([df1, df2], ignore_index=True)  ##combine both test and train data set to fill the missing data in the set with the appropriate values
cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
data = data[cols] #after combining data rearrange columns order
#print(data.describe())
#print(data)
data_Fare_of_embarked_equal_s_and_pclass_equal_3 = data[(data['Pclass']==3) & (data['Embarked']== 2)].filter(items=["Fare"]) #find fare col of embarked = s and pclass = 3
median_Fare = data_Fare_of_embarked_equal_s_and_pclass_equal_3.median() #get the madian value of col fare and replace nan value with it
data['Fare'][1043] = median_Fare
data["Fare"] = data["Fare"].astype(int) # from float to int
#print(data.isnull().sum()) ##get all nan numbers for each col

##########################handle Age missing data #########################################################

median_Age = data["Age"].median()
data.Age.fillna(median_Age, inplace=True) #fill Age with median age
#print(data[data['Age'].isnull()])
#print(data.isnull().sum()) ##get all nan numbers for each col
data['Age'] = data["Age"].astype(int)

############################## handel sex value #####################################################

data.loc[data["Sex"] == 'male', 'Sex'] = 0
data.loc[data["Sex"] == 'female', 'Sex'] = 1

############################## add new feature family size ################################################

FamilySize = ["Single"] * 1309
data["FamilySize"] = FamilySize
#print(data.groupby("FamilySize").size())
FamilySize1 = [1] * 1309
data["FamilySize1"] = FamilySize1
data['FamilySize1'] = data['Parch'] + data['SibSp'] + 1
#print(data.groupby("FamilySize1").size()) #too many i will reduce it to 3 groups single = 0 - large = 2 - small = 1
data.loc[(data['FamilySize1'] == 2 ) | (data['FamilySize1'] == 3) | (data['FamilySize1'] == 4), 'FamilySize'] = 'Small'
data.loc[(data['FamilySize1'] == 5 ) | (data['FamilySize1'] == 6) | (data['FamilySize1'] == 7) | (data['FamilySize1'] == 8)
         | (data['FamilySize1'] == 11), 'FamilySize'] = 'Large'
data = data.drop('FamilySize1', 1)
#print( data)
Family = [0] * 1309
data ["Family"] = Family
data.loc[data["FamilySize"] == 'Single', 'Family'] = 0
data.loc[data["FamilySize"] == 'Small', 'Family'] = 1
data.loc[data["FamilySize"] == 'Large', 'Family'] = 2
data = data.drop('FamilySize', 1)

#time to plot family vs survived

plot_Family_vs_Survived = data[["Family", "Survived"]].groupby(["Family"], as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=plot_Family_vs_Survived, order=[0, 1, 2], palette= "Set3")
plt.savefig("Family_Survived.png") #single and small family with code 0 and 1 has much more survival rate than large family size

############################################ Name handle and add title #########################################################################3

Title = ['Mr'] * 1309
data["Title"] = Title
data.loc[data['Name'].str.contains("Mrs."), 'Title'] = 'Mrs'
data.loc[data['Name'].str.contains("Miss. "), 'Title'] = 'Miss'
data.loc[data['Name'].str.contains("Capt. "), 'Title'] = 'Capt'
data.loc[data['Name'].str.contains("Col. "), 'Title'] = 'Col'
data.loc[data['Name'].str.contains("Don. "), 'Title'] = 'Don'
data.loc[data['Name'].str.contains("Dona. "), 'Title'] = 'Dona'
data.loc[data['Name'].str.contains("Dr. "), 'Title'] = 'Dr'
data.loc[data['Name'].str.contains("Jonkheer. "), 'Title'] = 'Jonkheer'
data.loc[data['Name'].str.contains("Lady. "), 'Title'] = 'Lady'
data.loc[data['Name'].str.contains("Major. "), 'Title'] = 'Major'
data.loc[data['Name'].str.contains("Master. "), 'Title'] = 'Master'
data.loc[data['Name'].str.contains("Mlle. "), 'Title'] = 'Mlle'
data.loc[data['Name'].str.contains("Mme. "), 'Title'] = 'Mme'
data.loc[data['Name'].str.contains("Ms. "), 'Title'] = 'Ms'
data.loc[data['Name'].str.contains("Rev. "), 'Title'] = 'Rev'
data.loc[data['Name'].str.contains("Sir. "), 'Title'] = 'Sir'
data.loc[data['Name'].str.contains("The Countess. "), 'Title'] = 'The Countess'
###################### reduce titles to master = 1 - miss = 2  - mr = 3 - mrs = 4 - other = 5 ####################################
data.loc[data['Title'].str.contains("Mme"), 'Title'] = 'Mrs'
data.loc[data['Title'].str.contains("Ms")| data['Title'].str.contains("Mlle"), 'Title'] = 'Miss'
data.loc[data['Title'].str.contains("Dona")| data['Title'].str.contains("Dr")| data['Title'].str.contains("Lady")|
         data['Title'].str.contains("Capt")| data['Title'].str.contains("Col")| data['Title'].str.contains("Don")|
         data['Title'].str.contains("Major")| data['Title'].str.contains("Rev")| data['Title'].str.contains("Sir")|
         data['Title'].str.contains("The Countess")| data['Title'].str.contains("Jonkheer"), 'Title'] = 'Other'

TitleNum = [3] * 1309
data ["TitleNum"] = TitleNum
data.loc[data["Title"] == 'Master', 'TitleNum'] = 1
data.loc[data["Title"] == 'Miss', 'TitleNum'] = 2
data.loc[data["Title"] == 'Mr', 'TitleNum'] = 3
data.loc[data["Title"] == 'Mrs', 'TitleNum'] = 4
data.loc[data["Title"] == 'Other', 'TitleNum'] = 5

data = data.drop('Title', 1)
plot_TitleNum_vs_Survived = data[["TitleNum", "Survived"]].groupby(["TitleNum"], as_index=False).mean()
sns.barplot(x='TitleNum', y='Survived', data=plot_TitleNum_vs_Survived, order=[1, 2, 3, 4, 5], palette= "Set3")
plt.savefig("TitleNum_Survived.png") #miss and mrs and master have higher survival rate than mr ond other. mr has lower survival rate

#################################### time to handle unwanted feat ################################################

data = data.drop(["PassengerId", "Name", "Ticket"], axis=1)
training_data = data[:891]
testing_data = data[891:]
PassengerId = test_data["PassengerId"]
##print(data)


##################################################################################################################

survived = training_data["Survived"]
training_data_without_survived = training_data.drop("Survived", axis = 1)
testdata_without_survived = testing_data.drop("Survived" , axis = 1) #drop nan from test data
X_Values = training_data_without_survived.values
m, n = X_Values.shape
survived_values = survived.values #find values of DF
X_Values_Test = testdata_without_survived.values
dimension_one_test, dimension_two_test = X_Values_Test.shape
x = np.ones((m, n+1))
x_test = np.ones(shape=(dimension_one_test, dimension_two_test+1))
x[:, -n:] = X_Values
x_test[:, -n:] = X_Values_Test
survived_values.shape=(m, 1)
y_test = np.ones(shape=(n+1, 1))
initial_theta = np.zeros(shape=(n+1, 1))

############## compute cost ####################################################################

c = cost(initial_theta,x,survived_values)
############# compute gradient ##################################################
gradient = np.zeros(len(initial_theta))
m_length = len(survived_values)
h = initial_theta.T.dot(x.T)
sigmoid_h = 1.0 / (1.0 + np.exp(-1.0 * h))
for j in range (len(initial_theta)):
    gradient[j] = (1.0/m_length)*sum((sigmoid_h.T-survived_values).T.dot(x[:, j]))

gradient.shape = (len(gradient), 1)

#################################################################################
result = op.minimize(fun= cost, x0=initial_theta, args=(x,survived_values), method='TNC')
opt_theta = result.x

array_opt = np.array(opt_theta)
a, b = x.shape
prediction = np.zeros(shape=(a, 1))
bb = x.dot(array_opt.T)
sigmoid_bb = 1.0 / (1.0 + np.exp(-1.0 * bb))
for i in range(0, sigmoid_bb.shape[0]):
    if sigmoid_bb[i] > 0.5:
        prediction[i] = 1
    else:
        prediction[i] = 0


t = (prediction == survived_values)
#print(np.mean(t)*100)

#################################time to predict test #######################################################

array_opt_y = np.array(opt_theta)
a_test, b_test = x_test.shape
prediction_test = np.zeros(shape=(a_test, 1))
k = x_test.dot(array_opt_y.T)
sigmoid_k = 1.0 / (1.0 + np.exp(-1.0 * k))
for i in range(0, sigmoid_k.shape[0]):
    if sigmoid_k[i] > 0.5:
        prediction_test[i] = 1
    else:
        prediction_test[i] = 0

prediction_test = prediction_test.astype(int)

output = pd.DataFrame({"PassengerId": PassengerId, "Survived": prediction_test.reshape(-1)})
header = ["PassengerId", "Survived"]
output.to_csv("final_prediction_result.csv", sep=",", columns=header, index=False)

print(output)

#print(gradient)


#print(opt_theta)
#print(initial_theta)
