from flask import Flask, redirect, render_template, request, session ,send_file ,url_for,make_response
from flask_session import Session
from flask_uploads import UploadSet, configure_uploads, ALL
from flask_socketio import SocketIO
import numpy as np
import pandas as pd
import pandas as df
from os import path, remove
from glob import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import io
import urllib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix




fig,ax=plt.subplots(figsize=(5,5))
ax=sns.set_style(style="darkgrid")



app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


csvdatafiles = UploadSet('data', ALL)
app.config['UPLOADED_DATA_DEST'] = 'static/data'
configure_uploads(app, csvdatafiles)


socketio = SocketIO(app)


#this is how we are getting the file that the user uploads. 
#then we are setting the path that we want to save it so we can use it later for predictions

@app.route('/')
def homeForm():
   session.clear()
   return render_template(['lina.html'])




@app.route('/prediction')
def prediction():
    session.clear()
    return render_template('prediction.html')   





@app.route('/prediction', methods=['POST','GET'])
def loadMain():
    # fixflask_uploads.UploadNotAllowed error
   try:
       maindata = 'static/data/' + csvdatafiles.save(request.files['maindata'])
   except Exception:
       maindata = 'static/data/' + str(request.form['maindata'])
   fulldata = readFile(maindata)
   #fulldata = pd.read_csv(maindata)
   checkflag = validate(maindata)
   session['maindata']=readFile(maindata)
   if checkflag[0] == True:
        
        
        results = fullAnalysis(maindata)
        allvars = session["fulldata"].columns.values
        cleanvars = session["normdata"].columns.values
        target = cleanvars[-1]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        newdf = fulldata.select_dtypes(include=numerics)
        list =[]
        list.extend(newdf)
        index_list =[]
        for feature in list:
           index_list.extend(Outliers(fulldata , feature))
        data_cleaned=remove1(fulldata,index_list)  

        
        new=fulldata.isnull().sum()

      
   if path.exists(maindata):
            remove(maindata)
        
        
            return render_template('prediction.html',data=fulldata )


@app.route('/regression', methods=['POST','GET'])
def reg():
        allvars = session["fulldata"].columns.values
        cleanvars = session["normdata"].columns.values
        
        return render_template('regression.html' ,  allvars=allvars, cleanvars=cleanvars  )

@app.route('/classification', methods=['POST','GET'])
def clas():
        allvars = session["fulldata"].columns.values
        cleanvars = session["normdata"].columns.values
        
        return render_template('classification.html' ,  allvars=allvars, cleanvars=cleanvars )


@app.route('/analysis', methods=['POST','GET'])
def ana():
       data=session['fulldata']
       datanew = data.select_dtypes(include=np.number) 
       mean=datanew.mean()
       cov=datanew.cov()
       plt.figure(figsize=(3,3))
       plt.legend()
       img = io.BytesIO()
       plt.boxplot(data)
       plt.savefig(img, format='png')

       img.seek(0)
        #plot_url = base64.b64encode(img.getvalue()).decode('utf8')
       box_data = urllib.parse.quote(base64.b64encode(img.read()).decode())



        
       return render_template('analysis1.html' ,  sumtables=contSummarize(data),box_url=box_data )        







@app.route('/resultregression',methods=['POST','GET'])
def regression():
    data = session["fulldata"]
    convert(data)
    if(request.method=='POST'):
         X,Y=request.form.getlist('X'),request.form.getlist('Y')
         x,y=data[X],data[Y]
         x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=100)
         data.dropna()
         regressor=LinearRegression()
         regressor.fit(x_train ,y_train)
         score=regressor.score(x_test,y_test)
         y_pred= regressor.predict(x_test)

         plt.figure(figsize=(5,5))
         plt.scatter(x_test,y_test,color='red')
         plt.plot(x_test, y_pred, color ='blue')
         plt.legend()
         img = io.BytesIO()
         plt.savefig(img, format='png')
        
         img.seek(0)
         #plot_url = base64.b64encode(img.getvalue()).decode('utf8')
         plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
         
         return render_template('regression.html' , predictiontable=table(y_pred),plot_url=plot_data)




@app.route('/resultclassification',methods=['POST','GET'])
def classification():
    data = session["fulldata"]
    convert(data)
    if(request.method=='POST'): 
         Z,C=request.form.getlist('Z'),request.form.getlist('C')
         c=data[C] 
         z=data[Z] 
         z_train,z_test,c_train,c_test=train_test_split(z,c,test_size=0.2)
         
         data.dropna()
         lr =  LogisticRegression()
         label_encoder = preprocessing.LabelEncoder()
         train_Y = label_encoder.fit_transform(c_train)
         model=lr.fit(z_train ,train_Y)
         y_pred=lr.predict(z_train)

         #test_calc = pd.concat([pd.DataFrame(train_Y).reset_index(drop=True),pd.DataFrame(y_pred).reset_index(drop=True)],axis=1)
         #test_calc.rename(columns={0: 'predicted'}, inplace=True)

         #test_calc['predicted'] = test_calc['predicted'].apply(lambda x: 1 if x > 0.5 else 0)
         df_table=confusion_matrix(train_Y,y_pred)
         
        

         #p = df_table[1,1] / (df_table[1,1] + df_table[0,1])
         accuracy= (df_table[0,0] + df_table[1,1]) / (df_table[0,0] + df_table[0,1] + df_table[1,0] + df_table[1,1])
         precision= df_table[1,1] / (df_table[1,1] + df_table[0,1])
         recall= df_table[1,1] / (df_table[1,1] + df_table[1,0])
         
         #r = df_table[1,1] / (df_table[1,1] + df_table[1,0])
         # print('f1 score: ', (2*p*r)/(p+r))
         
         format(model.score(z_test,c_test))
         f,w,e=roc_curve(c_test,model.predict_proba(z_test)[:,1])
         logit_roc_auc=roc_auc_score(c_test,model.predict(z_test))
         plt.figure(figsize=(6,5))

         plt.plot(f,w,label='Logistic Regression(area=%0.2f)' %logit_roc_auc)
         plt.plot([0,1],[0,1],'r--')
         plt.xlim([0.0,1.0])
         plt.ylim([0.0,1.05])
         plt.xlabel('false')
         plt.ylabel('true')
         plt.legend(loc="lower right")
         img = io.BytesIO()
         plt.savefig(img, format='png')
         img.seek(0)
         plot_url = base64.b64encode(img.getvalue()).decode('utf8')
         plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())

       
         
         #model1.fit(z_train ,train_Y)
         #score=model1.score(z_test,c_test)
         #y_prediction= model1.predict(z_test)
         return render_template('classification.html',plot_urlc=plot_data,accuracy=accuracy)




def remove1(data,ls):
    ls=sorted(set(ls))
    data=data.drop(ls)
    return data

def Outliers(data,ft):
    Q1 = data[ft].quantile(0.25)
    Q3 = data[ft].quantile(0.75)
    IQR = Q3-Q1
    lower_bound= Q1 -1.5 * IQR
    upper_bound= Q3 +1.5* IQR

    ls=data.index[ (data[ft]< lower_bound) | (data[ft]> upper_bound) ]

    return ls

def convert(data):
    for col_name in data.columns:
        if(data[col_name].dtype == 'object'):
           data[col_name]= data[col_name].astype('category')
           data[col_name] = data[col_name].cat.codes
    return data[col_name] 

def table(data):
   
    sum1 = pd.DataFrame(data,columns=['predictions'])
    sum2= sum1.to_html(classes='table',table_id='table')
    #sum2 = duplicated.to_html(classes='sumcont2')
    summaries = [sum2]
    return summaries

      

def contSummarize(fulldata):
    summary =  fulldata.describe(include = 'all').T
    #summary['nulls'] = fulldata.isna().sum().values
   # dtypes = []
   # for k in fulldata.columns.values:
    #    dtypes.append(fulldata[k].dtype)
    #summary['dType'] = dtypes
    #sum1 = summary[(summary['dType'] == 'float64') | (summary['dType'] == 'int64')].dropna(axis='columns')
    sum2 = summary.to_html(classes='table')
    summaries = [sum2]
    return summaries

def predictiontable(fulldata):
    summary =  fulldata.describe(include = 'all').T
    #summary['nulls'] = fulldata.isna().sum().values
    #dtypes = []
    #for k in fulldata.columns.values:
     #   dtypes.append(fulldata[k].dtype)
    #summary['dType'] = dtypes
    #sum1 = summary[(summary['dType'] == 'float64') | (summary['dType'] == 'int64')].dropna(axis='columns')
    sum2 = summary.to_html(classes='table',table_id='salma')
    summaries = [sum2]
    return summaries
         
def dup(data):
    duplicate=data[data.duplicated()].count().T
    sum2 = duplicate.to_html(classes='table')
    summaries = [sum2]
    return summaries

def clearCache():
    
    datafiles = glob('static/data/*')
    for d in datafiles:
        remove(d)


def validate(maindata):
    # change to methods to accomadate diff inputs
    checkflagcsv = False
    filetype = ''
    try:
        dfcheck = pd.read_csv(maindata)
        filetype = 'csv'
        targetvar = dfcheck.columns.values[-1]
        if (len(dfcheck.columns.values) > 1):
            # try numeric conversion?
            checkflagcsv = True
            # make diff error page for this?
    except Exception:
        checkflagcsv = False
       
    checkflag = [checkflagcsv, filetype]
    return checkflag


#def futureCheck(maindata, future):
    # do more diligence here (ex. same data types instead of names then rename. check order, etc)
 #   checkflag = False
  #  main = pd.read_csv(maindata)
   # fut = pd.read_csv(future) 
    #maincols = main.columns.values
    #futcols = fut.columns.values
    #if (np.array_equal((maincols[:-1]),futcols)):
     #   checkflag = True
    #else:
     #   checkflag = False
    #return checkflag


def fullAnalysis(maindata):
    fulldata = readFile(maindata)
    
    cleandata = cleanColumns(fulldata)
    # make diff error pages with the same template and link them (tar var, not enough data)
    # outdata = outliers(cleandata) - optional <remove outliers>
    normdata = normalize(cleandata)
    howcleanedvars = revealClean()
    results = [howcleanedvars]
    return results

def readFile(maindata):
    # clean string for common errors
    fulldata = pd.read_csv(maindata)
    # check file type and read diff types
    if (fulldata.columns.values[0] == 'Unnamed: 0'):
        fulldata = fulldata.drop(['Unnamed: 0'], axis=1)
    # add logic to check if col 1 is an index then fulldata = fulldata.set_index(fulldata.columns.values[0])?
    return fulldata

def cleanColumns(fulldata):
    # Clean target var too
    session["fulldata"] = fulldata
    cleandata = fulldata.copy()
    cleandata = cleandata.sort_index(axis = 0) 
    varnames = cleandata.columns.values
    targetvar = varnames[-1]
    targetdata = cleandata[targetvar].values
    cleandata = cleandata.drop([targetvar], axis=1)
    nacols = cleandata.isna().sum().values
    session["nullvar"] = []
    for k in range(len(nacols)):
        if (nacols[k]/len(cleandata) > .3):
            dropvar = varnames[k]
            cleandata = cleandata.drop([dropvar], axis=1)
            session["nullvar"].append(dropvar)
            # do some fillna here instead? (if # of cols dropped here is too much)
    session["datetimes"] = []
    session["smallDisc"] = []
    session["medDisc"] = []
    session["bigDisc"] = []
    for varname in cleandata.columns.values:
        if(cleandata[varname].dtype != np.float64 and cleandata[varname].dtype != np.int64):
            try:
                cleandata[varname] = pd.to_numeric(cleandata[varname])
            except Exception:
                cleandata = discClean(cleandata, varname)
    cleandata = cleandata.sort_index(axis = 1) 
    cleandata[targetvar] = targetdata
    cleandata = cleandata.dropna()
    session["tarDisc"] = []
    #cleandata = cleanTarget(cleandata, targetvar)
    # if data is too big and date is empty shuffle and head it (aka replace 5000 )
    if (len(cleandata.columns.values) > 10):
        cleandata = cleandata.head(5000)
    return cleandata


def discClean(cleandata, varname):
    # do other checks to ensure data is a datetime
    try:
        cleandata[varname] =  pd.to_datetime(cleandata[varname])
        cleandata = datetimeDisc(cleandata, varname)
    except Exception:
        # instead of size, change it to nominal vs ordinal
        univals = len(cleandata[varname].unique())
        # change upeer bound of .2 based on research 
        if ((univals < 2) or (univals/len(cleandata) > .2)) and (len(cleandata.columns.values) > 2):
            cleandata = bigDisc(cleandata, varname)
        elif (univals < 21):
            cleandata = smallDisc(cleandata, varname)
        else:
            cleandata = medDisc(cleandata, varname)
    return cleandata


def datetimeDisc(cleandata, varname):
    cleandata[varname + '_year'] = cleandata[varname].dt.year
    cleandata[varname + '_month'] = cleandata[varname].dt.month
    cleandata[varname + '_day'] = cleandata[varname].dt.day
    # add others? (ex. season, day of year, day of week?)
    cleandata = cleandata.drop([varname], axis=1)
    session["datetimes"].append(varname)
    return cleandata


def bigDisc(cleandata, varname):
    cleandata = cleandata.drop([varname], axis=1)
    # do some fillna here? (if # of cols dropped here is too much)
    session["bigDisc"].append(varname)
    return cleandata


def medDisc(cleandata, varname):
    # weight of evidence encoding?
    # mean encoding?
    # https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
    # save mappings to apply to future data
    univals = np.sort(cleandata[[varname]].dropna()[varname].unique())
    arbindex = (range(len(univals)))
    valdict = dict(zip(univals, arbindex))
    cleandata[varname] = cleandata[varname].replace(valdict)
    session["medDisc"].append([varname,univals])
    return cleandata


def smallDisc(cleandata, varname):
    univals = np.sort(cleandata[[varname]].dropna()[varname].unique())
    dummies = pd.get_dummies(cleandata[varname])
    dummies = dummies.sort_index(axis = 1)
    dummies.columns = [varname + '_' + str(col) for col in dummies.columns]
    cleandata = pd.concat([cleandata, dummies], axis=1)
    cleandata = cleandata.drop([varname], axis=1)
    session["smallDisc"].append([varname,univals])
    return cleandata


#def cleanTarget(cleandata, targetvar):
 #   if(cleandata[targetvar].dtype != np.float64 and cleandata[targetvar].dtype != np.int64):
  #      try:
   #         cleandata[targetvar] = pd.to_numeric(cleandata[targetvar])
    #    except Exception:
     #       cleandata = targetMap(cleandata, targetvar)
    #return cleandata


#def targetMap(cleandata, targetvar):
 #   univals = np.sort(cleandata[[targetvar]].dropna()[targetvar].unique())
  #  arbindex = (range(len(univals)))
   # valdict = dict(zip(univals, arbindex))
    #cleandata[targetvar] = cleandata[targetvar].replace(valdict)
    #session["tarDisc"] = [[targetvar,valdict]]
    #return cleandata


def revealClean():
    cleanedvars = []
    for k in session["datetimes"]:
        cleanedvars.append(str(k) + ' was converted into [' + str(k) + '_year, ' + str(k) + '_month, ' + str(k) + '_day].')
    for k in session["smallDisc"]:
        cleanedvars.append(str(k[0]) + ' was converted into ' + str(k[1]).replace("'", "").replace(" ", ", ") + '.')
    for k in session["medDisc"]:
        cleanedvars.append(str(k[0]) + ' was converted into a mapping of numbers.')
    for k in session["bigDisc"]:
        cleanedvars.append(str(k) + ' was dropped for being uninformative.')
    for k in session["nullvar"]:
        cleanedvars.append(str(k) + ' was dropped for having too many nulls.')
    for k in session["tarDisc"]:
        cleanedvars.append(str(k[0]) + ' was mapped on to: ' + str(k[1]))
    return cleanedvars


def cleanFuture(futuredata):
    cleandata = futuredata.copy()
    # replace with method calls
    for k in session["datetimes"]:
        cleandata[k] =  pd.to_datetime(cleandata[k], errors='coerce')
        cleandata[k + '_year'] = cleandata[k].dt.year
        cleandata[k + '_month'] = cleandata[k].dt.month
        cleandata[k + '_day'] = cleandata[k].dt.day
        cleandata = cleandata.drop([k], axis=1)
        valdict = dict(zip(univals, arbindex))
    for k in session["smallDisc"]:
        for dum in k[1]:
            newvar = k[0] + '_' + str(dum)
            cleandata[newvar] = 0
            cleandata[newvar] = np.where((cleandata[k[0]] == dum), 1, 0)
        cleandata = cleandata.drop([k[0]], axis=1)
    for k in session["medDisc"]:
        univals = k[1]
        arbindex = (range(len(univals)))
        cleandata[k] = cleandata[k].replace(valdict)
    for k in session["bigDisc"]:
        cleandata = cleandata.drop([k], axis=1)
    for k in session["nullvar"]:
        cleandata = cleandata.drop([k], axis=1)
    cleandata = cleandata.dropna()
    cleandata = cleandata.sort_index(axis = 1) 
    return cleandata


def normalize(cleandata):
    normdata = cleandata.copy()
    allcols = list(normdata.columns)
    allcols = allcols[:-1]
    for col in allcols:
        normdata[col] = (normdata[col] - normdata[col].mean())/normdata[col].std(ddof=0)
    normdata = normdata.dropna(axis='columns')
    session["normdata"] = normdata
    return normdata






@socketio.on('disconnect')
def disconnect_user():
    Flask.ext.login.logout_user()
    clearCache()
    session.clear()


if __name__ == '__main__':
    # write comments for everything later
    app.debug = True
    app.run()
    