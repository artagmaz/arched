#!/usr/bin/env python
# coding: utf-8

# # ArcheD correlation analysis

# In[24]:


# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statistics 


# In[2]:


x = '20_10_22'


# In[3]:


#Model
hist=np.load(x+'/abeta_history_'+x+'.npy',allow_pickle='TRUE').item()
stat=np.load(x+'/abeta_stat_'+x+'.npy',allow_pickle='TRUE')
prediction_test = np.load(x+'/abeta_prediction_'+x+'.npy',allow_pickle='TRUE')
prediction_train = np.load(x+'/abeta_train_prediction_'+x+'.npy',allow_pickle='TRUE')
prediction_all = np.load(x+'/abeta_all_prediction_'+x+'.npy',allow_pickle='TRUE')
test_label = np.load(x+'/abeta_testlab_'+x+'.npy',allow_pickle='TRUE')

info = pd.read_csv(x+'/abeta_info_improved_'+x+'.csv')



# In[4]:


#BEST Model
stat_best=np.load(x+'/abeta_stat_best_'+x+'.npy',allow_pickle='TRUE')
prediction_test_best = np.load(x+'/abeta_prediction_best_'+x+'.npy',allow_pickle='TRUE')
prediction_train_best = np.load(x+'/abeta_train_prediction_best_'+x+'.npy',allow_pickle='TRUE')
prediction_all_best = np.load(x+'/abeta_all_prediction_best_'+x+'.npy',allow_pickle='TRUE')
#dense_best = np.load(x+'/abeta_dense_out_best_'+x+'.npy',allow_pickle='TRUE')
#dense_all_best = np.load(x+'/abeta_dense_all_out_best_'+x+'.npy',allow_pickle='TRUE')


# In[6]:


#SUVR
av45 = pd.read_csv('/csc/epitkane/home/atagmazi/AD_DL_ADNI/PET_Image_Analysis/UCBERKELEYAV45_11_16_21.csv')
fbb = pd.read_csv('/csc/epitkane/home/atagmazi/AD_DL_ADNI/PET_Image_Analysis/UCBERKELEYFBB_11_16_21.csv')

suvr = pd.concat([av45, fbb], axis=0)


# In[7]:


#memory test
mem = pd.read_csv('/csc/epitkane/home/atagmazi/AD_DL_ADNI/NEUROBAT.csv')


# In[8]:


mem


# In[9]:


# finding AVLT 1-5 test score
avtot = mem[['AVTOT1','AVTOT2','AVTOT3','AVTOT4','AVTOT5']].sum(axis=1)


# In[10]:


mem[['AVTOT1','AVTOT2','AVTOT3','AVTOT4','AVTOT5']]


# In[11]:


# select only needed memory tests' results
mem_sel = mem[['RID','VISCODE','LIMMTOTAL','AVDEL30MIN','LDELTOTAL']]
mem_sel = pd.concat([mem_sel,avtot.rename('AVTOTSUM')],axis = 1)
mem_sel.loc[mem_sel['AVTOTSUM'] < 1 ,'AVTOTSUM'] = np.NaN
mem_sel


# In[12]:


num = prediction_all_best.shape[0]

i = info.loc[:num-1,:].copy()

i['prediction'] = prediction_all_best.flatten()
i['ABETA']= np.log(i['ABETA'])
i['TAU']= np.log(i['TAU'])
i['PTAU']= np.log(i['PTAU'])

#i = i.iloc[info.shape[0]-prediction_test_best.shape[0]:,:] #only test dataset


# In[13]:


num


# In[14]:


i['Subject.ID'].unique().shape


# In[18]:


main =  pd.merge(i, av45[['RID','VISCODE','SUMMARYSUVR_WHOLECEREBNORM','WHOLECEREBELLUM_SUVR']], #,'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF'
                 left_on=['RID','VISITCODE'], right_on = ['RID','VISCODE'], 
                 how = 'left').drop(columns= ['VISCODE'])

main =  pd.merge(main, fbb[['RID','VISCODE','SUMMARYSUVR_WHOLECEREBNORM','WHOLECEREBELLUM_SUVR']], #,'SUMMARYSUVR_WHOLECEREBNORM_1.08CUTOFF'
                 left_on=['RID','VISITCODE'], right_on = ['RID','VISCODE'],
                 how = 'left').drop(columns= ['VISCODE'])

main =  pd.merge(main, mem_sel,
                 left_on=['RID','VISITCODE'], right_on = ['RID','VISCODE'],
                 how = 'left').drop(columns= ['VISCODE'])


# In[19]:


main


# In[20]:


main.columns


# In[21]:


# keep only interesting for us columns (amyloid CSF, model prediction, SUVR and memory tests' score)
main_sel = main[['ABETA','prediction','SUMMARYSUVR_WHOLECEREBNORM_x','SUMMARYSUVR_WHOLECEREBNORM_y', 'LIMMTOTAL',
       'LDELTOTAL', 'AVTOTSUM','AVDEL30MIN' ]]


# In[22]:


main_sel


# In[25]:


# plot the correlation between selected parameters for all samples and classes separately 
fig, axes = plt.subplots(8,6,figsize = (20,18))

for i in range(0,8):
    axes[i,0].scatter(main_sel['ABETA'],main_sel.iloc[:, [i]].values,alpha=0.4, s = 15) 
    mask =~np.isnan(main_sel.iloc[:, [i]].values.flatten())
    s = stats.linregress(main_sel[mask]['ABETA'],main_sel.iloc[mask, [i]].values.flatten()) 
    axes[i,0].plot(main_sel['ABETA'], s[0]*main_sel['ABETA'] + s[1],linewidth=2, c = 'orange')
    axes[i,0].tick_params(axis='both', which='major', labelsize=13)
    axes[i,0].set_xlim(5,9)
    
for i in range(0,8):
    axes[i,1].scatter(main_sel['prediction'],main_sel.iloc[:, [i]].values,alpha=0.4, s = 15)
    mask =~np.isnan(main_sel.iloc[:, [i]].values.flatten())
    s = stats.linregress(main_sel[mask]['prediction'],main_sel.iloc[mask, [i]].values.flatten()) 
    axes[i,1].plot(main_sel['prediction'], s[0]*main_sel['prediction'] + s[1],linewidth=2, c = 'orange')
    axes[i,1].tick_params(axis='both', which='major', labelsize=13)
    axes[i,1].set_xlim(5,9)
    
for i in range(0,8):
    axes[i,2].scatter(main_sel[main['Research.Group']== 'AD']['prediction'],
                   main_sel[main['Research.Group']== 'AD'].iloc[:, [i]].values,alpha=0.4, s = 15) 
    mask =~np.isnan(main_sel[main['Research.Group']== 'AD'].iloc[:, [i]].values.flatten())
    s = stats.linregress(main_sel[main['Research.Group']== 'AD']['prediction'][mask],main_sel[main['Research.Group']== 'AD'].iloc[mask, [i]].values.flatten()) 
    axes[i,2].plot(main_sel[main['Research.Group']== 'AD']['prediction'], s[0]*
                   main_sel[main['Research.Group']== 'AD']['prediction'] + s[1],linewidth=2, c = 'orange')
    axes[i,2].tick_params(axis='both', which='major', labelsize=13)
    axes[i,2].set_xlim(5,9)
    
for i in range(0,8):
    axes[i,3].scatter(main_sel[main['Research.Group'].isin(['MCI','LMCI','EMCI'])]['prediction'],
                   main_sel[main['Research.Group'].isin(['MCI','LMCI','EMCI'])].iloc[:, [i]].values,alpha=0.4, s = 15) 
    mask =~np.isnan(main_sel[main['Research.Group'].isin(['MCI','LMCI','EMCI'])].iloc[:, [i]].values.flatten())
    s = stats.linregress(main_sel[main['Research.Group'].isin(['MCI','LMCI','EMCI'])]['prediction'][mask],main_sel[main['Research.Group'].isin(['MCI','LMCI','EMCI'])].iloc[mask, [i]].values.flatten()) 
    axes[i,3].plot(main_sel[main['Research.Group'].isin(['MCI','LMCI','EMCI'])]['prediction'], s[0]*
                   main_sel[main['Research.Group'].isin(['MCI','LMCI','EMCI'])]['prediction'] + s[1],
                   linewidth=2, c = 'orange')
    axes[i,3].tick_params(axis='both', which='major', labelsize=13)
    axes[i,3].set_xlim(5,9)
    
for i in range(0,8):
    axes[i,4].scatter(main_sel[main['Research.Group']== 'CN']['prediction'],
                   main_sel[main['Research.Group']== 'CN'].iloc[:, [i]].values,alpha=0.4, s = 15) 
    mask =~np.isnan(main_sel[main['Research.Group']== 'CN'].iloc[:, [i]].values.flatten())
    s = stats.linregress(main_sel[main['Research.Group']== 'CN']['prediction'][mask],main_sel[main['Research.Group']== 'CN'].iloc[mask, [i]].values.flatten()) 
    axes[i,4].plot(main_sel[main['Research.Group']== 'CN']['prediction'], s[0]*
                   main_sel[main['Research.Group']== 'CN']['prediction'] + s[1],linewidth=2, c = 'orange')
    axes[i,4].tick_params(axis='both', which='major', labelsize=13)
    axes[i,4].set_xlim(5,9)
    
for i in range(0,8):
    axes[i,5].scatter(main_sel[main['Research.Group']== 'SMC']['prediction'],
                   main_sel[main['Research.Group']== 'SMC'].iloc[:, [i]].values,alpha=0.4, s = 15) 
    if i != 3:
        mask =~np.isnan(main_sel[main['Research.Group']== 'SMC'].iloc[:, [i]].values.flatten())
        s = stats.linregress(main_sel[main['Research.Group']== 'SMC']['prediction'][mask],main_sel[main['Research.Group']== 'SMC'].iloc[mask, [i]].values.flatten()) 
        axes[i,5].plot(main_sel[main['Research.Group']== 'SMC']['prediction'], s[0]*
                       main_sel[main['Research.Group']== 'SMC']['prediction'] + s[1],linewidth=2, c = 'orange')
    axes[i,5].tick_params(axis='both', which='major', labelsize=13)
    axes[i,5].set_xlim(5,9)
    
axes[0,0].set_ylabel('Clinical Aβ CSF',size = 15)
axes[1,0].set_ylabel('Model predicted\n Aβ CSF',size = 15)
axes[2,0].set_ylabel('AV45 SUVR',size = 15)
axes[3,0].set_ylabel('FBB SUVR',size = 15)
axes[4,0].set_ylabel('LMI',size = 15)
axes[5,0].set_ylabel('LMD',size = 15)
axes[6,0].set_ylabel('AVLT 1-5',size = 15)
axes[7,0].set_ylabel('AVLTdel',size = 15)

axes[7,0].set_xlabel('Clinical Aβ CSF',size = 15)
axes[7,1].set_xlabel('Predicted Aβ CSF',size = 15)
axes[7,2].set_xlabel('Predicted Aβ CSF',size = 15)
axes[7,3].set_xlabel('Predicted Aβ CSF',size = 15)
axes[7,4].set_xlabel('Predicted Aβ CSF',size = 15)
axes[7,5].set_xlabel('Predicted Aβ CSF',size = 15)

axes[0,0].set_title('All samples',size = 20)
axes[0,1].set_title('All samples',size = 20)
axes[0,2].set_title('AD',size = 20)
axes[0,3].set_title('MCI',size = 20)
axes[0,4].set_title('CN',size = 20)
axes[0,5].set_title('SMC',size = 20)

plt.savefig(x+'/correlation_scatterplot_'+x+'.png')
plt.savefig(x+'/correlation_scatterplot_'+x+'.svg', format="svg")


# In[26]:


# correlation matrix for all parameters
cor_main = main.corr() #Pearson


# In[27]:


cor_main


# In[28]:


cor_main = cor_main.drop(columns =['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'],
             index = ['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'])


# In[29]:


summary = pd.DataFrame()


# In[30]:


# correlation for our model prediction all samples with all parameters
summary['all_pred'] = cor_main['prediction']
# correlation for amyloid CSF all samples with all parameters
summary['all_real'] = cor_main['ABETA']


# In[31]:


summary


# In[32]:


# correlation for CSF amyloid and model prediction separately for clinical subclasses
m=main[main['Research.Group']=='AD']
cor_m = m.corr()
cor_m= cor_m.drop(columns =['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'],
             index = ['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'])
summary['AD_pred'] = cor_m['prediction']
summary['AD_real'] = cor_m['ABETA']
print(len(m.index))

m=main[main['Research.Group'].isin(['MCI','LMCI','EMCI'])]
cor_m = m.corr()
cor_m= cor_m.drop(columns =['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'],
             index = ['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'])
summary['MCI total_pred'] = cor_m['prediction']
summary['MCI total_real'] = cor_m['ABETA']
print(len(m.index))

m=main[main['Research.Group']=='LMCI']
cor_m = m.corr()
cor_m= cor_m.drop(columns =['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'],
             index = ['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'])
summary['LMCI_pred'] = cor_m['prediction']
summary['LMCI_real'] = cor_m['ABETA']
print(len(m.index))

m=main[main['Research.Group']=='EMCI']
cor_m = m.corr()
cor_m= cor_m.drop(columns =['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'],
             index = ['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'])
summary['EMCI_pred'] = cor_m['prediction']
summary['EMCI_real'] = cor_m['ABETA']
print(len(m.index))

m=main[main['Research.Group']=='SMC']
cor_m = m.corr()
cor_m= cor_m.drop(columns =['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'],
             index = ['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'])
summary['SMC_pred'] = cor_m['prediction']
summary['SMC_real'] = cor_m['ABETA']
print(len(m.index))

m=main[main['Research.Group']=='CN']
cor_m = m.corr()
cor_m= cor_m.drop(columns =['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'],
             index = ['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2','RID'])
summary['CN_pred'] = cor_m['prediction']
summary['CN_real'] = cor_m['ABETA']
print(len(m.index))


# In[33]:


summary


# In[34]:


#m = main[['prediction','LDELTOTAL','Research.Group']].dropna()
#stats.pearsonr(m['Research.Group'=='AD']['prediction'],m['LDELTOTAL'])


# In[35]:


# preparing correlation matrix plot
summary = summary.rename(index={'SUMMARYSUVR_WHOLECEREBNORM_x': 'SUVR_AV45',
                                'SUMMARYSUVR_WHOLECEREBNORM_y': 'SUVR_FBB'})


# In[36]:


summary.loc[['ABETA','SUVR_AV45','SUVR_FBB','LIMMTOTAL','LDELTOTAL','AVTOTSUM','AVDEL30MIN'],
            ['all_pred','all_real','AD_pred','MCI total_pred', 'SMC_pred','CN_pred']]


# In[37]:


vis = summary.loc[['ABETA','SUVR_AV45','SUVR_FBB','LIMMTOTAL','LDELTOTAL','AVTOTSUM','AVDEL30MIN'],
            ['all_pred','all_real','AD_pred','MCI total_pred', 'SMC_pred','CN_pred']]


# In[38]:


vis.T


# In[39]:


vis = vis.T.rename(index={'all_pred': 'All samples \npredected Aβ CSF \nmeasures \n(1868)',
                          'all_real': 'All samples \nclinical Aβ \nCSF measures \n(1868)',
                         'AD_pred':'AD predicted \nAβ CSF (190)',
                         'MCI total_pred':'MCI predicted \nAβ CSF (928)',
                         'SMC_pred':'SMC predicted \nAβ CSF (144)',
                         'CN_pred':'CN predicted \nAβ CSF (606)'},
                columns = {'ABETA': 'Clinical Aβ \nCSF measures'})


# In[40]:


vis


# In[41]:


plt.figure(figsize = (19,10))
sns.set(font_scale = 2)

g = sns.heatmap(vis, annot=True, cmap= 'coolwarm')

g.set_xticklabels(vis.columns, rotation=40, size = 20)
g.set_yticklabels(vis.index, rotation=0, size = 20)
plt.tight_layout()
#plt.savefig("corr.png")

plt.savefig("corr.svg", format="svg")


# In[42]:



 
print(statistics.stdev([-0.558806,-0.672508]))
print(statistics.mean([-0.558806,-0.672508]))


# In[43]:


stats.pearsonr(main['ABETA'],main['prediction'])


# In[44]:


# correlation and p-value for CSF amyloid vs SUVR and memory tests
m = main[['ABETA','SUMMARYSUVR_WHOLECEREBNORM_x']].dropna()
print(stats.pearsonr(m['ABETA'],m['SUMMARYSUVR_WHOLECEREBNORM_x']))

m = main[['ABETA','SUMMARYSUVR_WHOLECEREBNORM_y']].dropna()
print(stats.pearsonr(m['ABETA'],m['SUMMARYSUVR_WHOLECEREBNORM_y']))

m = main[['ABETA','LIMMTOTAL']].dropna()
print(stats.pearsonr(m['ABETA'],m['LIMMTOTAL']))

m = main[['ABETA','LDELTOTAL']].dropna()
print(stats.pearsonr(m['ABETA'],m['LDELTOTAL']))

m = main[['ABETA','AVTOTSUM']].dropna()
print(stats.pearsonr(m['ABETA'],m['AVTOTSUM']))

m = main[['ABETA','AVDEL30MIN']].dropna()
print(stats.pearsonr(m['ABETA'],m['AVDEL30MIN']))


# In[47]:


pval = pd.DataFrame()


# In[48]:


# correlation and p-value for all groups
col = ['ABETA','SUMMARYSUVR_WHOLECEREBNORM_x','SUMMARYSUVR_WHOLECEREBNORM_y','LIMMTOTAL','LDELTOTAL',
      'AVTOTSUM','AVDEL30MIN']
for i in range(7):
    m = main[['prediction',col[i],'Research.Group']].dropna()
    print('all', col[i], stats.pearsonr(m['prediction'],m[col[i]]))
    pval['all '+col[i]] = [stats.pearsonr(m['prediction'],m[col[i]])[1]]
    
    if i>0:
        m_cl = main[['ABETA',col[i],'Research.Group']].dropna()
        print('all clinical', col[i], stats.pearsonr(m_cl['ABETA'],m_cl[col[i]]))
        pval['all clinical '+col[i]] = [stats.pearsonr(m_cl['ABETA'],m_cl[col[i]])[1]]
    
    mad = m[m['Research.Group']=='AD']
    print('AD', col[i], stats.pearsonr(mad['prediction'],mad[col[i]]))
    pval['AD '+col[i]] = [stats.pearsonr(mad['prediction'],mad[col[i]])[1]]
    
    mmci = m[m['Research.Group'].isin(['MCI','LMCI','EMCI'])]
    print('MCI total', col[i], stats.pearsonr(mmci['prediction'],mmci[col[i]]))
    pval['MCI total '+col[i]] = [stats.pearsonr(mmci['prediction'],mmci[col[i]])[1]]
    
    mlmci = m[m['Research.Group']=='LMCI']
    if len(mlmci.index)>0:
        print('LMCI', col[i], stats.pearsonr(mlmci['prediction'],mlmci[col[i]]))
    
    memci = m[m['Research.Group']=='EMCI']
    if len(memci.index)>0:
        print('EMCI', col[i], stats.pearsonr(memci['prediction'],memci[col[i]]))
    
    msmc = m[m['Research.Group']=='SMC']
    if len(msmc.index)>0:
        print('SMC', col[i], stats.pearsonr(msmc['prediction'],msmc[col[i]]))
        pval['SMC '+col[i]] = [stats.pearsonr(msmc['prediction'],msmc[col[i]])[1]]
    
    mcn = m[m['Research.Group']=='CN']
    print('CN', col[i], stats.pearsonr(mcn['prediction'],mcn[col[i]]))
    pval['CN '+col[i]] = [stats.pearsonr(mcn['prediction'],mcn[col[i]])[1]]


# In[49]:


#applying FDR correction on p-value
p = pval.T


# In[50]:


from statsmodels.sandbox.stats.multicomp import multipletests


# In[51]:


p['qvalue'] = multipletests(p[0], alpha=0.05, method='fdr_bh')[1]


# In[52]:


p


# In[53]:


p<= 0.01


# In[ ]:




