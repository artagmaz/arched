#######################################################
#### Processing data from ADNI, compilling into metafile, training models
#### for rescaling fluid biomarker measures from INNO-BIA AlzBio3 system to Roche Elecsys #####
#### Tagmazian Arina

#### set directory ####
setwd('/Users/tagmarin/artagmaz/ad_dl/adni/')
#### upload libraries ####
library(tidyr)
library(ggplot2)
library(gridExtra)
library(moments)
library(dplyr)
#### upload data ####

# csv from ADNI search result
data = read.csv('idaSearch_all_16_04_2022.csv')

#biomarkers measures
biomarker = read.csv('UPENNBIOMK_MASTER.csv')
biomarker_elecsys  = read.csv('UPENNBIOMK12_01_04_21.csv')
biomarker_elecsys2017  = read.csv('UPENNBIOMK9_04_19_17.csv')
biomarker_elecsys2019  = read.csv('UPENNBIOMK10_07_29_19.csv')

visit_dict = read.csv('VISITS.csv')

#blind-friendly colors
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


#### data transformation ####

# split Description into 2 columns
data2 = separate(data, Description,into = c('pipeline','Description'),sep = ' <-' )
# delete duplicates
r = grep('repeat|Repeat|REPEAT', data2$Description)
data2 = data2[-r,]
# modifying name of FDG scans
data2$pipeline[data2$pipeline == 'Coreg, Avg, Standardized Image and Voxel Size'] = 
  'FDG Coreg, Avg, Standardized Image and Voxel Size'
data2$pipeline[data2$pipeline == 'Co-registered, Averaged'] = 
  'FDG Co-registered, Averaged'
data2$pipeline[data2$pipeline == 'Coreg, Avg, Std Img and Vox Siz, Uniform Resolution'] = 
  'FDG Coreg, Avg, Std Img and Vox Siz, Uniform Resolution'
data2$pipeline[data2$pipeline == 'Co-registered Dynamic'] = 'FDG Co-registered Dynamic'
# total number of samples
length(unique(data2$Subject.ID))  

#### normalization pipeline summary ####
# type of processing for MRI and PET images
x = table(data2$pipeline[data2$Modality == 'MRI'])
x2 = x[order(x, decreasing = T)]
ind = c()
for(i in 1:length(x2)){
  t = data2[data2$pipeline == names(x2)[i] & data2$Modality == 'MRI', ]
  ind[i] = length(unique(t$Subject.ID))
}
x3 = cbind(x2,ind)

write.csv(x3, 'mri_norm_02_04_2022.csv')

y = table(data2$pipeline[data2$Modality == 'PET'])
y2 = y[order(y, decreasing = T)]
ind2 = c()
for(i in 1:length(y2)){
  t = data2[data2$pipeline == names(y2)[i] & data2$Modality == 'PET', ]
  ind2[i] = length(unique(t$Subject.ID))
}
y3 = cbind(y2,ind2)

write.csv(y3, 'pet_norm_02_04_2022.csv')


#### METADATA file creation ####
meta = cbind(Image_id = paste0('I',data2$Image.ID),
             RID = as.integer(sapply(strsplit(as.character(data2$Subject.ID), split = '_'),"[",3)),
             data2[,1:8],
             VISITCODE = c(0),
             data2[,9:15],
             pet_tracer = c(NA),
             csf_method = c(NA),
             ABETA = c(NA),
             TAU = c(NA),
             PTAU = c(NA),
             CLASS = c(NA),
             PATH = c(NA))

# add PET tracer type
meta$pet_tracer[meta$Modality == 'PET'] = sapply(strsplit(meta$pipeline[meta$Modality == 'PET'], split = ' '),"[",1)
  
# add VISITCODE
visit_dict2 = visit_dict
visit_dict2$VISNAME = gsub('Continuing Pt','Cont Pt',visit_dict$VISNAME)
visit_dict2$VISNAME = gsub(' - ','-',visit_dict2$VISNAME)

for (i in 1:nrow(visit_dict2)){
  n = grep(paste0(visit_dict2$VISNAME[i],'$'), meta$Visit)
  meta$VISITCODE[n] = as.character(visit_dict2$VISCODE[i])
}
  
table(meta$VISITCODE)


# add biomarkers values
biomarker_med = biomarker[biomarker$BATCH == 'MEDIAN',]
temp = sapply(strsplit(as.character(biomarker_elecsys2017$COMMENT), split = '= '),"[",2)
temp = as.numeric(sapply(strsplit(as.character(temp), split = ' pg/mL'),"[",1))
temp = temp[!is.na(temp)]

biomarker_elecsys2017$ABETA = as.character(biomarker_elecsys2017$ABETA)
g = grep('>1700',biomarker_elecsys2017$ABETA)
biomarker_elecsys2017$ABETA[g]  = as.numeric(temp)

  
colnames(biomarker_elecsys2019)[7] = 'ABETA'
colnames(biomarker_elecsys2017)[5] = 'PROT'



biomarker_common = rbind(cbind(biomarker_med[,c(1,2,8,9,10)], PROT = c('ADNI'), method = 'AlzBio3'),
                         cbind(biomarker_elecsys[,c(1,2,8,10,9,5)], method = 'Elecsys'),
                         cbind(biomarker_elecsys2017[,c(1,2,10,11,12,5)], method = 'Elecsys'),
                         cbind(biomarker_elecsys2019[,c(1,2,7,9,8)],PROT = c('ADNI 3'), method = 'Elecsys')) 

biomarker_common = biomarker_common[order(biomarker_common$RID),]
biomarker_common[,3] = as.numeric(biomarker_common[,3])
biomarker_common[,4] = as.numeric(biomarker_common[,4])
biomarker_common[,5] = as.numeric(biomarker_common[,5])

# put biomarker values into meta file
for(i in 1:nrow(biomarker_common)){
  meta$ABETA[meta$RID == biomarker_common$RID[i] &
               meta$VISITCODE == biomarker_common$VISCODE[i]] = biomarker_common$ABETA[i]
  meta$TAU[meta$RID == biomarker_common$RID[i] &
               meta$VISITCODE == biomarker_common$VISCODE[i]] = biomarker_common$TAU[i]
  meta$PTAU[meta$RID == biomarker_common$RID[i] &
               meta$VISITCODE == biomarker_common$VISCODE[i]] = biomarker_common$PTAU[i]
  meta$csf_method[meta$RID == biomarker_common$RID[i] &
              meta$VISITCODE == biomarker_common$VISCODE[i]] = as.character(biomarker_common$method[i])
}

table(is.na(meta$ABETA))
table(is.na(meta$ABETA[meta$Modality == 'PET']))
table(is.na(meta$ABETA[meta$Modality == 'MRI']))

## define classes based on the threshold values
#0 A-T-
#  1 A-T+ 
#  2 A+T-
#  3 A+T+
  


meta$CLASS[meta$ABETA > 192 & meta$PTAU < 23 & meta$csf_method == 'AlzBio3'] = 0
meta$CLASS[meta$ABETA > 192 & meta$PTAU >= 23 & meta$csf_method == 'AlzBio3'] = 1
meta$CLASS[meta$ABETA <= 192 & meta$PTAU < 23 & meta$csf_method == 'AlzBio3'] = 2
meta$CLASS[meta$ABETA <= 192 & meta$PTAU >= 23 & meta$csf_method == 'AlzBio3'] = 3

meta$CLASS[meta$ABETA > 980 & meta$PTAU < 21.8 & meta$csf_method == 'Elecsys'] = 0
meta$CLASS[meta$ABETA > 980 & meta$PTAU >= 21.8 & meta$csf_method == 'Elecsys'] = 1
meta$CLASS[meta$ABETA <= 980 & meta$PTAU < 21.8 & meta$csf_method == 'Elecsys'] = 2
meta$CLASS[meta$ABETA <= 980 & meta$PTAU >= 21.8 & meta$csf_method == 'Elecsys'] = 3

table(meta$CLASS)
table(meta$Research.Group[meta$CLASS == 0])

#### images with and without csf values ####
summary = matrix(ncol = 2, nrow = 2)
rownames(summary) = rownames(table(meta$Modality))
colnames(summary) = c('with CSF', 'without CSF')

for(i in 1:2){
  x = meta[meta$Modality == rownames(summary)[i],]
  summary[i,1] = sum(!is.na(x$ABETA))
  summary[i,2] = sum(is.na(x$ABETA))
}
summary

#### plots research group/Class ~ age distribution ####

rg_age = data.frame(id = meta$Subject.ID,Research.Group = meta$Research.Group, Age = meta$Age)
rg_age = unique(rg_age)
table(rg_age$Research.Group)
rg_age = rg_age[rg_age$Age > 1,]
rg_age$Research.Group[rg_age$Research.Group %in% c('EMCI','LMCI','MCI')] = 'MCI'
png('research_group_aging_dist.png')
p <- ggplot(rg_age, aes(x=Research.Group, y=Age, color=Research.Group)) + 
  geom_violin() +
  geom_boxplot(width=0.1) +
  scale_fill_brewer(palette="Dark2")
p
dev.off()

png('res_group_class_aging_dist.png')
meta_cl = meta[,c('Subject.ID','Research.Group','CLASS','Age')]
meta_cl = unique(meta_cl)
meta_cl = meta_cl[!is.na(meta_cl$CLASS),]
meta_cl$Research.Group[meta_cl$Research.Group %in% c('EMCI','LMCI','MCI')] = 'MCI'

summ =meta_cl %>%
  group_by(CLASS, Research.Group) %>%
  summarise(count=n(), Age = max(Age)+2)

p <- ggplot(meta_cl, aes(x=Research.Group, y=Age, fill = Research.Group )) + 
  geom_violin() +
  geom_boxplot(width=0.1) +
  scale_fill_brewer(palette="Dark2")+
  geom_text(aes(x=Research.Group, y=Age,label = count), data = summ)+
  facet_wrap(~CLASS,labeller = labeller(CLASS = c('0'='A-T-','1'='A-T+','2'='A+T-','3'='A+T+'))) 
  

p
dev.off()

png('class_aging_dist.png')
p <- ggplot(meta_cl, aes(x=CLASS, y=Age,group = CLASS, color = CLASS )) + 
  geom_violin() +
  geom_boxplot(width=0.1) +
  scale_fill_brewer(palette="Dark2")

p
dev.off()
###

length(unique(meta$RID[meta$Modality == 'PET' & !is.na(meta$ABETA)])) 

#### rescaling models ####
biomarker_elecsys2017$VISCODE = as.character(biomarker_elecsys2017$VISCODE)
# combine old and new approaches in one table
bion_old_new = data.frame(RID = biomarker_med$RID, VISCODE =  biomarker_med$VISCODE,
                          abeta_old = biomarker_med$ABETA,
                          tau_old = biomarker_med$TAU, ptau_old = biomarker_med$PTAU,
                          abeta_2017 = c(NA), tau_2017 = c(NA), ptau_2017 = c(NA), stat = c('train'))


bion_old_new$stat = as.character(bion_old_new$stat)


for(i in 1:nrow(biomarker_elecsys2017)){
  bion_old_new$abeta_2017[bion_old_new$RID == biomarker_elecsys2017$RID[i] &
                            bion_old_new$VISCODE == biomarker_elecsys2017$VISCODE[i]] = as.character(biomarker_elecsys2017$ABETA[i])
  bion_old_new$tau_2017[bion_old_new$RID == biomarker_elecsys2017$RID[i] &
                          bion_old_new$VISCODE == biomarker_elecsys2017$VISCODE[i]] = as.character(biomarker_elecsys2017$TAU[i])
  bion_old_new$ptau_2017[bion_old_new$RID == biomarker_elecsys2017$RID[i] &
                           bion_old_new$VISCODE == biomarker_elecsys2017$VISCODE[i]] = as.character(biomarker_elecsys2017$PTAU[i])
  
}

for(i in 3:8){
  bion_old_new[,i]= as.numeric(bion_old_new[,i])
}

bion_old_new$tau_2017[bion_old_new[,7]<3] = NA
bion_old_new$ptau_2017[bion_old_new[,8]<3] = NA

####TRAIN MODEL NEW CODE####
png('lr_csf_sig_03_10_22.png', width = 10, height = 6, units= "in", res = 200)

svg('lr_csf_sig_03_10_22.svg',   
    width = 20, height = 7)
bion_old_new$stat[is.na(bion_old_new$tau_2017)] = 'predicted'

bio_predicted = bion_old_new[bion_old_new$stat=='predicted',]
bio_fitting = bion_old_new[bion_old_new$stat!='predicted',]
bio_train = bio_fitting[1:round(nrow(bio_fitting)*0.8),]
bio_test = bio_fitting[(round(nrow(bio_fitting)*0.8)+1):nrow(bio_fitting),]

###
par(oma = c(4,1,1,1),mfrow = c(1,3), mar = c(5, 5, 4, 2))

old = c(3,4,5)
treshold_old = c(192,93,23)

new = c(6,7,8)
treshold_new = c(980,245,21.8)

name = c('Abeta','Tau','Ptau')


r2_test = matrix(ncol = 3, nrow = 3)
colnames(r2_test) = c('abeta','tau','ptau')
rownames(r2_test) = c('lm','polynomial','sigmoid')

r2_train = matrix(ncol = 3, nrow = 3)
colnames(r2_train) = c('abeta','tau','ptau')
rownames(r2_train) = c('lm','polynomial','sigmoid')

accuracy = matrix(ncol = 3, nrow = 3)
colnames(accuracy) = c('abeta train/test','tau train/test', 'ptau train/test')
rownames(accuracy) = c('lm','polynomial','sigmoid')

for(i in 1:3){
  o = old[i]
  n = new[i]
  
  test = bio_test[!is.na(bio_test[,o]),]
  
  x = bio_train[,o]
  y = bio_train[,n]
  
  b = boxplot(x, plot = F)
  x[x %in% b$out] = NA
  b = boxplot(y, plot = F)
  y[y %in% b$out] = NA
  
  x2 = x[!is.na(x) & !is.na(y)]
  y2 = y[!is.na(x) & !is.na(y)]
  
  model = lm(y2~x2)
  model2 = lm(y2 ~ poly(x2,3))
  
  # IT IS CORRECT, https://stackoverflow.com/questions/33033176/using-r-to-fit-a-sigmoidal-curve 
  model3 = nls(y2 ~ SSlogis(x2, Asym, xmid, scal), data=data.frame(x2 = x2, y2 = y2))
  
  #Plot of model prediction curve for train dataset
  predicted.intervals <- predict(model2,as.data.frame(x2),interval='confidence',
                                 level=0.99)
  predicted.intervals3 <- predict(model3,as.data.frame(x2),interval='confidence',
                                  level=0.99)
  
  plot(x2,y2, main = name[i], xlab = 'AlzBio3', ylab = 'Roche Elecsys', 
       cex.lab = 3, cex.main = 2, cex.axis = 1.5, pch = 20)
  points(bio_test[,o], bio_test[,n], col='#009E73', pch = 19)
  abline(model,col = cbPalette[2], lwd=3)
  
  ix <- sort(x2,index.return=T)$ix
  lines(x2[ix], predicted.intervals[ix], col=cbPalette[3], lwd=3 )
  #lines(x2[ix], predicted.intervals3[ix], col=cbPalette[4], lwd=3 )
  
  abline(v = treshold_old[i], col = 'grey', lwd=2)
  abline(h = treshold_new[i], col = 'grey', lwd=2)
  
  #Predict test dataset 
  
  model_pred = predict(model,newdata = data.frame(x2=test[,o]))
  model_pred2 = predict(model2,newdata = data.frame(x2=test[,o]))
  model_pred3 = predict(model3,newdata = data.frame(x2=test[,o]))
  
  r2_train[1,i] = paste(summary(model)$r.squared,cor(y2,x2, method = 'pearson'), sep = ';')
  r2_train[2,i] = paste(summary(model2)$r.squared,cor(y2,x2, method = 'pearson'), sep = ';')
  
  RSS.p <- sum(residuals(model3)^2)
  TSS <- sum((y2 - mean(y2))^2)
  r2_train[3,i] = paste(1 - (RSS.p/TSS),cor(y2,x2, method = 'pearson'), sep = ';')
  
  r2_test[1,i] = paste(cor(model_pred,test[,n], method = 'pearson')^2,cor(model_pred,test[,n], method = 'pearson'), sep = ';')
  r2_test[2,i] = paste(cor(model_pred2,test[,n], method = 'pearson')^2,cor(model_pred2,test[,n], method = 'pearson'), sep = ';')
  r2_test[3,i] = paste(cor(model_pred3,test[,n], method = 'pearson')^2,cor(model_pred3,test[,n], method = 'pearson'), sep = ';')
  
  
  #rmse[1,i] = paste(sqrt(mean(model$residuals^2)),
  #             sqrt(sum((model_pred - test[,n])^2)/length(test[,n])), sep = ';')
  #rmse[2,i] = paste(sqrt(mean(model2$residuals^2)),
  #                  sqrt(sum((model_pred2 - test[,n])^2)/length(test[,n])), sep = ';')
  #rmse[3,i] = paste(sqrt(mean(model3$residuals^2)),
  #                  sqrt(sum((model_pred3 - test[,n])^2)/length(test[,n])), sep = ';')
  
  ac = function(j,model_x, data){
    acc = c()
    #bion_old_new_t$stat = 'train'
    o = old[j]
    n = new[j]
    #bion_old_new_t$stat[is.na(bion_old_new_t[,n])] = 'test'
    
    data$prediction = predict(model_x,data.frame(x2=data[,o]))
    
    if(j == 1){
      data$cl_initial = as.numeric(data[,n]) < treshold_new[j]
      data$cl_pred = as.numeric(data$prediction) < treshold_new[j]
      
      acc_tab = matrix(nco = 2, nrow = 2)
      colnames(acc_tab) = c('real_pos','false_pos')
      rownames(acc_tab) = c('false_neg','real_neg')
      acc_tab[1,1] = nrow(data[data$cl_initial == T & data$cl_pred == T,])
      acc_tab[1,2] = nrow(data[data$cl_initial == F & data$cl_pred == T,])
      acc_tab[2,1] = nrow(data[data$cl_initial == T & data$cl_pred == F,])
      acc_tab[2,2] = nrow(data[data$cl_initial == F & data$cl_pred == F,])
      
      #acc[1,i] = acc_tab[1,1]/(acc_tab[1,1]+acc_tab[2,1])
      #acc[2,i] = acc_tab[1,2]/(acc_tab[1,2]+acc_tab[2,2])
      
      acc = (acc_tab[1,1]+acc_tab[2,2])/sum(acc_tab)
    }
    else{
      data$cl_initial = as.numeric(data[,n]) > treshold_new[j]
      data$cl_pred = as.numeric(data$prediction) > treshold_new[j]
      
      acc_tab = matrix(nco = 2, nrow = 2)
      colnames(acc_tab) = c('real_pos','false_pos')
      rownames(acc_tab) = c('false_neg','real_neg')
      acc_tab[1,1] = nrow(data[data$cl_initial == T & data$cl_pred == T,])
      acc_tab[1,2] = nrow(data[data$cl_initial == F & data$cl_pred == T,])
      acc_tab[2,1] = nrow(data[data$cl_initial == T & data$cl_pred == F,])
      acc_tab[2,2] = nrow(data[data$cl_initial == F & data$cl_pred == F,])
      
      #acc[1,i] = acc_tab[1,1]/(acc_tab[1,1]+acc_tab[2,1])
      #acc[2,i] = acc_tab[1,2]/(acc_tab[1,2]+acc_tab[2,2])
      acc = (acc_tab[1,1]+acc_tab[2,2])/sum(acc_tab)
    }
    return(acc)
  }
  accuracy[1,i] = paste(ac(i,model,bio_train),ac(i,model,test), sep = ';')

  accuracy[2,i] = paste(ac(i,model2,bio_train),ac(i,model2,test), sep = ';')

  accuracy[3,i] = paste(ac(i,model3,bio_train),ac(i,model3,test), sep = ';')
  
  bion_old_new[bion_old_new$stat == 'predicted',n] = predict(model2,data.frame(x2 = bion_old_new[bion_old_new$stat == 'predicted',o]))
  if (i ==1){
    abeta_linear = model
    abeta_polynomial = model2
    abeta_ix = ix
    abeta_predint = predicted.intervals
  }
  
}



par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = 'l', bty = 'n', xaxt = 'n', yaxt = 'n')
legend('bottom',legend = c('linear regression','polynomial regression'), 
       col = c(cbPalette[2],cbPalette[3]), lwd = 6, xpd = TRUE, horiz = TRUE, cex = 1.5, seg.len=1, bty = 'n')
# xpd = TRUE makes the legend plot to the figure


dev.off()

write.csv(r2,'r2_sigmoid.csv')



#### final meta file; replace alzbio3 values to elecsys ####

meta_final = meta
for(i in 1:nrow(bion_old_new)){
  meta_final$ABETA[meta_final$RID == bion_old_new$RID[i] &
                     meta_final$VISITCODE == bion_old_new$VISCODE[i] & 
                     meta_final$csf_method == 'AlzBio3'] = bion_old_new$abeta_2017[i]
  
  meta_final$TAU[meta_final$RID == bion_old_new$RID[i] & 
                   meta_final$VISITCODE == bion_old_new$VISCODE[i] &
                   meta_final$csf_method == 'AlzBio3'] = bion_old_new$tau_2017[i]
  meta_final$PTAU[meta_final$RID == bion_old_new$RID[i] &
                    meta_final$VISITCODE == bion_old_new$VISCODE[i] &
                    meta_final$csf_method == 'AlzBio3'] = bion_old_new$ptau_2017[i]
  meta_final$csf_method[meta_final$RID == bion_old_new$RID[i] &
                          meta_final$VISITCODE == bion_old_new$VISCODE[i] & 
                          meta_final$csf_method == 'AlzBio3'] = 'Elecsys_predicted'
}

meta_final$CLASS[meta_final$ABETA > 980 & meta_final$PTAU < 21.8] = 0
meta_final$CLASS[meta_final$ABETA > 980 & meta_final$PTAU >= 21.8] = 1
meta_final$CLASS[meta_final$ABETA <= 980 & meta_final$PTAU < 21.8] = 2
meta_final$CLASS[meta_final$ABETA <= 980 & meta_final$PTAU >= 21.8] = 3


## save final metafile
write.csv(meta_final,'metafile_adni_all_12_04_2023.csv')


png('res_group_class_aging_dist.png')
x = meta_final[meta_final$pet_tracer %in% c('AV45','FBB','PIB'),]
meta_cl = x[,c('Subject.ID','Research.Group','CLASS','Age')]
meta_cl = unique(meta_cl)
meta_cl = meta_cl[!is.na(meta_cl$CLASS),]
meta_cl$Research.Group[meta_cl$Research.Group %in% c('EMCI','LMCI','MCI')] = 'MCI'

summ =meta_cl %>%
  group_by(CLASS, Research.Group) %>%
  summarise(count=n(), Age = max(Age)+2)

p <- ggplot(meta_cl, aes(x=Research.Group, y=Age, fill = Research.Group )) + 
  geom_violin() +
  geom_boxplot(width=0.1) +
  scale_fill_brewer(palette="Dark2")+
  geom_text(aes(x=Research.Group, y=Age,label = count), data = summ)+
  facet_wrap(~CLASS,labeller = labeller(CLASS = c('0'='A-T-','1'='A-T+','2'='A+T-','3'='A+T+'))) 


p
