#!/usr/bin/env python3
# VISIT WITH US - WELLNESS TOURISM PACKAGE PREDICTION
# Run in Colab: !python tourism_package_mlops.py --data /content/tourism.csv
# Outputs: train/test splits, trained model, reports, Streamlit app and GitHub Actions workflow.

import argparse, json, shutil, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
EXPECTED = ['CustomerID','ProdTaken','Age','TypeofContact','CityTier','DurationOfPitch',
'Occupation','Gender','NumberOfPersonVisiting','NumberOfFollowups','ProductPitched',
'PreferredPropertyStar','MaritalStatus','NumberOfTrips','Passport','PitchSatisfactionScore',
'OwnCar','NumberOfChildrenVisiting','Designation','MonthlyIncome']
TARGET, SEED = 'ProdTaken', 42

def obs(text):
    print('\n' + '='*18 + ' OBSERVATION ' + '='*18)
    print(text)
    print('='*50 + '\n')

def main():
    import pandas as pd, numpy as np, joblib
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

    parser=argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--root', default='tourism_mlops_project')
    args=parser.parse_args()
    root=Path(args.root).resolve()
    for folder in ['data','artifacts','deployment','reports','.github/workflows']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    this_script=Path(__file__).resolve()
    if this_script != (root/'tourism_package_mlops.py').resolve():
        shutil.copy2(this_script,root/'tourism_package_mlops.py')

    candidates=[Path(args.data)] if args.data else []
    candidates += [Path('tourism.csv'),Path('/content/tourism.csv'),Path('/mnt/data/tourism.csv')]
    source=next((p.resolve() for p in candidates if p.exists()),None)
    if source is None:
        try:
            from google.colab import files
            print('Please upload tourism.csv')
            uploaded=files.upload()
            source=Path(next(iter(uploaded))).resolve()
        except Exception:
            raise FileNotFoundError('Place tourism.csv beside this script or pass --data /path/tourism.csv')
    csv_path=root/'data/tourism.csv'
    if source != csv_path.resolve(): shutil.copy2(source,csv_path)
    df=pd.read_csv(csv_path)

    missing=sorted(set(EXPECTED)-set(df.columns))
    if missing: raise ValueError(f'Missing expected columns: {missing}')
    if df[TARGET].nunique()<2: raise ValueError('ProdTaken must contain both 0 and 1.')
    print('DATA REGISTRATION')
    print('Shape:',df.shape,'\nTarget counts:\n',df[TARGET].value_counts().sort_index())
    obs(f'Dataset registered with {len(df):,} rows and {df.shape[1]} columns. All expected fields are present. Review target counts above to understand class balance.')

    # Clean duplicates, remove rows without target and drop record identifiers.
    before=len(df)
    df=df.drop_duplicates().dropna(subset=[TARGET]).copy()
    id_cols=[c for c in df.columns if c.lower().startswith('unnamed') or c=='CustomerID']
    df=df.drop(columns=id_cols,errors='ignore')
    df[TARGET]=pd.to_numeric(df[TARGET],errors='coerce')
    df=df.dropna(subset=[TARGET])
    df[TARGET]=df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0,1}): raise ValueError('ProdTaken should contain only 0 and 1.')
    X,y=df.drop(columns=TARGET),df[TARGET]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=SEED,stratify=y)
    train=X_train.copy(); train[TARGET]=y_train
    test=X_test.copy(); test[TARGET]=y_test
    train.to_csv(root/'artifacts/train.csv',index=False)
    test.to_csv(root/'artifacts/test.csv',index=False)
    print('PREPARATION: before=',before,'after=',len(df),'dropped identifiers=',id_cols)
    print('Train shape:',train.shape,'Test shape:',test.shape)
    obs(f'After removing duplicates and rows without a target, {len(df):,} records remain. An 80:20 stratified split created {len(train):,} training and {len(test):,} test rows. Identifier columns {id_cols} were removed. Missing predictor values are imputed within the model pipeline.')

    numeric=X_train.select_dtypes(include='number').columns.tolist()
    categorical=X_train.select_dtypes(exclude='number').columns.tolist()
    prep=ColumnTransformer([
      ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
      ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical)
    ])
    candidates={
      'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED),
      'Random Forest':RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=SEED,n_jobs=1),
      'Extra Trees':ExtraTreesClassifier(n_estimators=250,class_weight='balanced',random_state=SEED,n_jobs=1)
    }
    rows=[]
    for name,est in candidates.items():
        pipe=Pipeline([('preprocess',prep),('model',est)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test); proba=pipe.predict_proba(X_test)[:,1]
        rows.append({'model':name,'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0),'roc_auc':roc_auc_score(y_test,proba)})
    comparison=pd.DataFrame(rows).sort_values(['roc_auc','f1'],ascending=False)
    comparison.to_csv(root/'reports/model_comparison.csv',index=False)
    print('BASELINE COMPARISON\n',comparison.to_string(index=False))
    bestrow=comparison.iloc[0]
    obs(f"{bestrow['model']} achieved the highest baseline ROC-AUC ({bestrow['roc_auc']:.3f}) on the held-out test data. Compare precision and recall as well: precision relates to efficient marketing outreach, while recall relates to how many actual purchasers are identified.")

    # Tune Random Forest using stratified cross-validation.
    pipe=Pipeline([('preprocess',prep),('model',RandomForestClassifier(class_weight='balanced',random_state=SEED,n_jobs=1))])
    grid={'model__n_estimators':[150,250],'model__max_depth':[None,12],'model__min_samples_split':[2,5],'model__min_samples_leaf':[1,2]}
    search=GridSearchCV(pipe,grid,scoring='roc_auc',cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED),n_jobs=1,verbose=1)
    search.fit(X_train,y_train)
    model=search.best_estimator_
    pred=model.predict(X_test); proba=model.predict_proba(X_test)[:,1]
    metrics={'best_model':'Tuned Random Forest','best_cv_roc_auc':float(search.best_score_),'test_accuracy':float(accuracy_score(y_test,pred)),'test_precision':float(precision_score(y_test,pred,zero_division=0)),'test_recall':float(recall_score(y_test,pred,zero_division=0)),'test_f1':float(f1_score(y_test,pred,zero_division=0)),'test_roc_auc':float(roc_auc_score(y_test,proba)),'best_params':search.best_params_,'confusion_matrix':confusion_matrix(y_test,pred).tolist(),'classification_report':classification_report(y_test,pred,output_dict=True,zero_division=0)}
    joblib.dump(model,root/'deployment/model.joblib')
    (root/'reports/metrics.json').write_text(json.dumps(metrics,indent=2))
    print('BEST PARAMETERS:',search.best_params_)
    for k in ['test_accuracy','test_precision','test_recall','test_f1','test_roc_auc']: print(k,round(metrics[k],4))
    print('Confusion matrix [[TN,FP],[FN,TP]]:\n',np.array(metrics['confusion_matrix']))
    obs(f"The tuned Random Forest test ROC-AUC is {metrics['test_roc_auc']:.3f}; F1 is {metrics['test_f1']:.3f}, precision is {metrics['test_precision']:.3f}, and recall is {metrics['test_recall']:.3f}. Use the confusion matrix to discuss false positives and missed buyers. These metrics describe this test split and should be checked again on newer data.")

    # Deployment application (model pipeline includes all preprocessing).
    app='''import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
st.set_page_config(page_title="Wellness Tourism Predictor",page_icon="✈️")
st.title("Wellness Tourism Package - Purchase Prediction")
st.write("Enter customer details to estimate purchase likelihood.")
@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent/"deployment"/"model.joblib")
model=load_model()
with st.form("customer"):
    a,b=st.columns(2)
    with a:
        Age=st.number_input("Age",18,100,35)
        TypeofContact=st.selectbox("Type of Contact",["Self Enquiry","Company Invited"])
        CityTier=st.selectbox("City Tier",[1,2,3])
        DurationOfPitch=st.number_input("Duration of Pitch",min_value=0.0,value=10.0)
        Occupation=st.selectbox("Occupation",["Salaried","Small Business","Large Business","Free Lancer"])
        Gender=st.selectbox("Gender",["Male","Female","Fe Male"])
        NumberOfPersonVisiting=st.number_input("Number of People Visiting",1,20,2)
        NumberOfFollowups=st.number_input("Number of Follow-ups",0,20,3)
        ProductPitched=st.selectbox("Product Pitched",["Basic","Deluxe","Standard","Super Deluxe","King"])
    with b:
        PreferredPropertyStar=st.selectbox("Preferred Property Star",[3,4,5])
        MaritalStatus=st.selectbox("Marital Status",["Single","Married","Divorced","Unmarried"])
        NumberOfTrips=st.number_input("Number of Trips",0.0,50.0,3.0)
        Passport=st.selectbox("Has Passport?",[0,1])
        PitchSatisfactionScore=st.selectbox("Pitch Satisfaction Score",[1,2,3,4,5])
        OwnCar=st.selectbox("Owns a Car?",[0,1])
        NumberOfChildrenVisiting=st.number_input("Number of Children Visiting",0,10,0)
        Designation=st.selectbox("Designation",["Executive","Manager","Senior Manager","AVP","VP"])
        MonthlyIncome=st.number_input("Monthly Income",min_value=0.0,value=20000.0)
    submitted=st.form_submit_button("Predict")
if submitted:
    row=pd.DataFrame([locals()[k] for k in []])
    row=pd.DataFrame([{"Age":Age,"TypeofContact":TypeofContact,"CityTier":CityTier,"DurationOfPitch":DurationOfPitch,"Occupation":Occupation,"Gender":Gender,"NumberOfPersonVisiting":NumberOfPersonVisiting,"NumberOfFollowups":NumberOfFollowups,"ProductPitched":ProductPitched,"PreferredPropertyStar":PreferredPropertyStar,"MaritalStatus":MaritalStatus,"NumberOfTrips":NumberOfTrips,"Passport":Passport,"PitchSatisfactionScore":PitchSatisfactionScore,"OwnCar":OwnCar,"NumberOfChildrenVisiting":NumberOfChildrenVisiting,"Designation":Designation,"MonthlyIncome":MonthlyIncome}])
    p=float(model.predict_proba(row)[0,1])
    st.metric("Estimated purchase probability",f"{p:.1%}")
    st.success("Potential purchaser (class 1)" if p>=.5 else "Less likely to purchase (class 0)")
    st.caption("Model estimate only; not a guarantee of customer behaviour.")
'''
    (root/'app.py').write_text(app)
    (root/'requirements.txt').write_text('pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\njoblib>=1.3\nstreamlit>=1.30\n')
    workflow='''name: Tourism Package Prediction CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: write
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python tourism_package_mlops.py --data data/tourism.csv --root .
      - name: Commit trained model and reports
        if: github.event_name != 'pull_request'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add deployment/model.joblib reports artifacts
          git diff --cached --quiet || git commit -m "Automated model training"
          git push
'''
    (root/'.github/workflows/pipeline.yml').write_text(workflow)
    (root/'README.md').write_text('# Visit with Us - Tourism Package Prediction\n\nRun: `python tourism_package_mlops.py --data data/tourism.csv`\n\nThe pipeline validates data, prepares stratified train/test splits, compares baseline classifiers, tunes a Random Forest, evaluates it, saves the fitted preprocessing/model pipeline, and creates a Streamlit app and GitHub Actions workflow.\n')
    print('\nPROJECT COMPLETE. Folder:',root)
    print('Model:',root/'deployment/model.joblib')
    print('App:',root/'app.py')
    print('Workflow:',root/'.github/workflows/pipeline.yml')
    obs('The run generated the model, metrics, data splits, Streamlit app, requirements file and GitHub Actions workflow. Review the metrics and validate the app inputs before deployment.')

if __name__=='__main__':
    main()
