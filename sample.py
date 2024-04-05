
import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
#with open(r'D:\\Capstone\\finalproject\\classification_model.pkl','rb') as f:
   #@ model = pickle.load(f)
modelname=r"D:\Capstone\finalproject\classification_randomforestclassifier.joblib"
pp=joblib.load(modelname)
print('8',modelname)

#model = pickle.load(open(r, 'rb'))

def run():
    #img1 = Image.open('bank.png')
    #img1 = img1.resize((156,145))
    #st.image(img1,use_column_width=False)
    st.title("NBFI Prediction using Machine Learning")

    ## Account No
    Client_Income = st.text_input('Client_Income')
    Car_Owned = st.text_input('Car_Owned')
    Bike_Owned=st.text_input('Bike_Owned')
    Active_Loan=st.text_input('Active_Loan')
    House_Own=st.text_input('House_Own')
    Child_Count=st.text_input('Child_Count')
    
    Credit_Amount=st.text_input('Credit_Amount')
    Loan_Annuity=st.text_input('Loan_Annuity')
    
    Accompany_Client=([1, 6, 4, 3, 5, 0, 2])
    accompany=list(range(len(Accompany_Client)))
    acc=st.selectbox('Accompany_Client',accompany,format_func=lambda x: Accompany_Client[x])

    Client_Income_Type=([1, 5, 4, 2, 8, 6, 7, 3, 0])
      
    Income_Type=list(range(len(Client_Income_Type)))
    cit=st.selectbox("Client_Income_Type",Income_Type, format_func=lambda x: Client_Income_Type[x])

    Client_Education=([4, 0, 1, 5, 2, 3])
    education=list(range(len(Client_Education)))
    ce= st.selectbox("Client_Education",education, format_func=lambda x: Client_Education[x])

    ## For gender
    Client_Marital_Status =([1, 3, 2, 4, 0])
    Marital_Status = list(range(len(Client_Marital_Status)))
    cms=st.selectbox("Client_Marital_status",Marital_Status, format_func=lambda x: Client_Marital_Status[x])

    

    
    Client_Gender=([1, 0, 3, 2])
    gender=list(range(len(Client_Gender)))
    cg=st.selectbox("Client_Education",gender, format_func=lambda x: Client_Gender[x])

    Loan_Contract_Type=([0, 1, 2])
    contract_type=list(range(len(Loan_Contract_Type)))
    lct= st.selectbox("Client_Education",contract_type, format_func=lambda x: Loan_Contract_Type[x])

    Client_Housing_Type=([1, 0, 3, 2, 6, 4, 5])
    Housing_Type=list(range(len(Client_Housing_Type)))
    ht=st.selectbox('Client_Housing_Type',Housing_Type,format_func=lambda x: Client_Housing_Type[x])

    Population_Region_Relative=st.text_input('Population_Region_Relative')
    Age_Days=st.text_input('Age_Days')
    Employed_Days=st.text_input('Employed_Days')
    Registration_Days=st.text_input('Registration_Days')
    ID_Days=st.text_input('ID_Days')
    Own_House_Age=st.text_input('Own_House_Age')
    Mobile_Tag=st.text_input('Mobile_Tag')
    Homephone_Tag=st.text_input('Homephone_Tag')
    Workphone_Working=st.text_input('Workphone_Working')

    Client_Occupation=([14, 18, 13,  8,  3,  4, 10,  0,  6,  1,  5, 17,  9, 11,  2, 12, 16,
        7, 15])
    occupation=list(range(len(Client_Occupation)))
    co=st.selectbox('Client_occupation',occupation,format_func=lambda x:Client_Occupation[x])
    Client_Family_Members=st.text_input('Client_Family_Members')
    Cleint_City_Rating=st.text_input('Cleint_City_Rating')
    Application_Process_Day=st.text_input('Application_Process_Day')
    Application_Process_Hour=st.text_input('Application_Process_Hour')
    Client_Permanent_Match_Tag=([1, 0])
    permanant_Match_Tag=list(range(len(Client_Permanent_Match_Tag)))
    cpmt=st.selectbox('Client_Permanent_Match_Tag',permanant_Match_Tag,format_func=lambda x:Client_Permanent_Match_Tag[x])
    Client_Contact_Work_Tag=([1, 0])
    contact_Match_Tag=list(range(len(Client_Contact_Work_Tag)))
    ccmt=st.selectbox('Client_Contact_Match_Tag',contact_Match_Tag,format_func=lambda x:Client_Contact_Work_Tag[x])

    #Client_Contact_Work_Tag=st.number_input('Client_Contact_Work_Tag')
    Type_Organization=([42,  5, 30, 57, 41, 16, 33, 47, 26, 55, 12,  7, 21, 40, 28, 46, 39,
       11, 20,  3, 38, 51,  6, 13, 54, 43, 24,  2,  1, 53,  4, 34, 10,  9,
       56, 52, 31,  8, 35, 36, 48, 50, 44, 32,  0, 19, 17, 14, 22, 18, 27,
       45, 29, 15, 23, 37, 49, 25])
    organization=list(range(len(Client_Occupation)))
    to=st.selectbox('Type_organization',organization,format_func=lambda x:Type_Organization[x])
    Score_Source_1=st.text_input('Score_Source_1')
    Score_Source_2=st.text_input('Score_Source_2')
    Score_Source_3=st.text_input('Score_Source_3')
    Social_Circle_Default=st.text_input('Social_Circle_Default')
    Phone_Change=st.text_input('Phone_Change')
    Credit_Bureau=st.text_input('Credit_Bureau')

    #
        #with open(r'D:\Capstone\final project\classification_model.pkl', 'rb') as f:
                #model = pickle.load(f)
               
    if st.button("Submit"):
      user_input=pd.DataFrame({'Client_Income':[Client_Income],
                           'Car_Owned':[Car_Owned],
                           'Bike_Owned':[Bike_Owned],
                           'Active_Loan':[Active_Loan],
                           'House_Own':[House_Own],
                           'Child_Count':[Child_Count],
                           'Credit_Amount':Credit_Amount,
                           'Loan_Annuity':[Loan_Annuity],
                           'Accompany_Client':[acc],
                           'Client_Income_Type':[cit],
                           'Client_Education':[ce],
                           'Client_Marital_Status':[cms],
                           'Client_Gender':[cg],
                           'Loan_Contract_Type':[lct],
                           'Client_Housing_Type':[ht],
                           'Population_Region_Relative':[Population_Region_Relative],
                           'Age_Days':[Age_Days],
                           'Employed_Days':[Employed_Days],
                           'Registration_Days':[Registration_Days],
                           'ID_Days':[ID_Days],
                           'Own_House_Age':[Own_House_Age],
                           'Mobile_Tag':[Mobile_Tag],
                           'Homephone_Tag':[Homephone_Tag],
                           'Workphone_Working':[Workphone_Working],
                           'Client_Occupation':[co],
                           'Client_Family_Members':[Client_Family_Members],
                           'Cleint_City_Rating':[Cleint_City_Rating],
                           'Application_Process_Day':[Application_Process_Day],
                           'Application_Process_Hour':[Application_Process_Hour],
                           'Client_Permanent_Match_Tag':[cpmt],
                           'Client_Contact_Work_Tag':[ccmt],
                           'Type_Organization':[to],
                           'Score_Source_1':[Score_Source_1],
                            'Score_Source_2':[Score_Source_2],
                           'Score_Source_3':[Score_Source_3],
                           'Social_Circle_Default':[Social_Circle_Default],
                           'Phone_Change':[Phone_Change],
                           'Credit_Bureau':[Credit_Bureau]})

      print(user_input)
      #st.write("Predicted:", prediction)

      y_p=pp.predict(user_input)
      st.write("Predicted:", y_p)
      if(y_p==0):
             st.write("No Default")
      else:
             st.write("deafault" )









     #features = [[acc,cit,ce,cms,cg,lct,ht,co,cpmt,ccmt,to]]
               
     #print(features)
     #print(model)
     #prediction = model.predict(features)
     #print(model)
     #lc = [str(i) for i in prediction]
     #ans = int("".join(lc))
     #if ans == 0:
              #print("default")
     #else:
             #print("no default")
            
            

run()