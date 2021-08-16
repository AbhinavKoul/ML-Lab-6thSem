weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
humidity=['High', 'High','High','High','Normal','Normal','Normal','High', 'Normal','Normal','Normal','High','Normal','High'] 
windy=['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak','strong']


play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

from sklearn import preprocessing

labelEncoder = preprocessing.LabelEncoder()

Weather_encoder = labelEncoder.fit_transform(weather)
Temp_encoder = labelEncoder.fit_transform(temp)
humidity_encoder = labelEncoder.fit_transform(humidity)
windy_encoder = labelEncoder.fit_transform(windy)

label = labelEncoder.fit_transform(play)

features = tuple(zip(Weather_encoder,Temp_encoder,humidity_encoder,windy_encoder))
print(features)

#import Guassian Naive Bayes Model
from sklearn.naive_bayes import GuassianNB

model = GuassianNB()
model.fit(features,label)

#checking output for new data1
test_data = [0,2,0,0]
predicted = model.predict([test_data])
if(predicted):
    print("Play Tennis: Yes")
else:
    print("Play Tennis: No")