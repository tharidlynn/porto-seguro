from sklearn.externals import joblib
import pandas as pd

from matplotlib import pyplot as plt

def save_features(train):
    # For plot importance only
    joblib.dump(train.drop(['id','target'], axis=1).columns.values, 'feature_names.pkl')

    print('Successfully dump feature_names for using in the future')

def load_features(model):
        '''
    For using with Stratified K fold only. you can use native train test just like above example
    '''

    # Use this snippet for return list
    # importance_df.sort_values('fscore', ascending=False)[:10].name.tolist()

    names = joblib.load('feature_names.pkl')
    # Map feature to name in dictionary format
    name_f = {}
    for i in range(len(names)):
        f = 'f' + str(i)
        name_f[f] = names[i]

    importance = model.get_fscore()
    importance_df = pd.DataFrame.from_dict(data=importance, orient='index')
    importance_df['f'] = importance_df.index
    importance_df['fscore'] = importance_df[0]

    importance_df.index = range(importance_df.shape[0])
    importance_df.drop(0, axis=1, inplace=True)
    importance_df['name'] = importance_df.f.map(name_f)

    importance_df.sort_values('fscore', ascending=False)

    # Plot
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(131)
    importance_df.sort_values('fscore').plot(kind='barh', x='name', y='fscore', legend=False, ax=ax)
    plt.title('Features importance')
    plt.xlabel('relative importance')
    plt.show()
