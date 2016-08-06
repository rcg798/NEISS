# before we answer the questions, 
# let's get the data all in one place

# first, we need to import the libraries we need
import pandas as pd
import numpy as np

# now let's read in the 4 csv files
# we can keep overwriting the location variable since we don't need it after the file is read in
location = r'/Users/rachaelgilbert/Desktop/BIHomework/NEISS2014.csv'
neiss = pd.read_csv(location)
location = r'/Users/rachaelgilbert/Desktop/BIHomework/Disposition.csv'
disposition = pd.read_csv(location)
location = r'/Users/rachaelgilbert/Desktop/BIHomework/BodyParts.csv'
body_parts = pd.read_csv(location)
location = r'/Users/rachaelgilbert/Desktop/BIHomework/DiagnosisCodes.csv'
diagnosis_codes = pd.read_csv(location)

# let's see what the dataframes look like
# this will give us a sense of how the data is structured

# here i use the head term for a preview
# but could also use e.g. print(neiss[:3]) 
# if we wanted to a certain number of rows in the preview

# ok so looks like we have our big dataset here
print("Neiss:")
print(neiss.head())

# and looks like these give the explanations for the neiss codes 
print("Disposition:")
print(disposition.head())
print("Body parts:")
print(body_parts.head())
print("Diagnosis codes:")
print(diagnosis_codes.head())

# let's join them all together to decode the main dataset codes 
# and have all the data in one place

# add in explanation of disposition code
neiss_all1 = pd.merge(neiss, disposition, left_on='disposition', right_on='Code')
# build on that and add in explanation of body parts code
neiss_all2 = pd.merge(neiss_all1, body_parts, left_on='body_part', right_on='Code')
# and lastly add on explanation of diagnosis code
neiss_all3 = pd.merge(neiss_all2, diagnosis_codes, left_on='diag', right_on='Code')

# how does it look now? excellent, data all in one place
print("Merging NEISS with code explanations:")
print(neiss_all3.head())

##############################
# What are the top three body parts most frequently represented in this dataset?
##############################

### ANSWER:
### Head
### Face
### Finger

# need to group by body part, then do a count of top 3
by_body_part = neiss_all3.groupby('BodyPart')
print("Top body parts most represented:")
print(by_body_part.size().sort_values(ascending=False)[:3])

##############################
# What are the top three body parts that are least frequently represented?
##############################

### ANSWER:
### If we are counting body parts as NEISS does:
### 25-50% of body
### Pubic region
### Not recorded
### If we are counting body parts more conventionally:
### Pubic region
### Internal organs
### Upper arm

# again group by body part, and count lowest 5
print("Top body parts least represented:")
print(by_body_part.size().sort_values(ascending=True)[:5])

##############################
# How many injuries in this dataset involve a skateboard?
##############################

### ANSWER:
### 466

# make a new column that flags narratives with a skateboard mention
neiss_all3['sb_flag'] = neiss_all3['narrative'].str.contains('.*skateboard.*', regex=True, case=False)

print("How many injuries involve a skateboard:")
print(len(neiss_all3.query('sb_flag == True')))

##############################
# Of those injuries, what percentage were male and what percentage were female?
##############################

### ANSWER:
### 17.6% are female and 82.4% are male

print("Skateboard injuries split by sex:")
# to make readable, I split these off into 3 vars
# but can also calc everything in-line (just a very long line)
total = neiss_all3[(neiss_all3['sb_flag'] == True)]
male = neiss_all3[(neiss_all3['sb_flag'] == True)&(neiss_all3['sex'] == 'Male')]
female = neiss_all3[(neiss_all3['sb_flag'] == True)&(neiss_all3['sex'] == 'Female')]
print(len(female)/len(total), "F injuries and", len(male)/len(total), "M injuries")

##############################
# What was the average age of someone injured in an incident involving a skateboard?
##############################

### ANSWER:
### Mean of 18 years old
### Median of 16 years old

# first we have to recode the ages, since everything above 200 is actually under 2 years old
# and we don't want that to incorrectly skew the mean
# though I would hope there aren't 2 year olds on skateboards!

# since over 200 ranges from 0 to 24 months, let's just call all that as 1 year old
neiss_all3['age2'] = np.where(neiss_all3['age']>=200, 1, neiss_all3['age'])

print("Skateboard injuries mean age:")
print(neiss_all3.query('sb_flag == True').age2.mean())
print("Skateboard injuries median age:")
print(neiss_all3.query('sb_flag == True').age2.median())

##############################
# What diagnosis had the highest hospitalization rate? 
##############################

### ANSWER:
### If we define hospitalization by just the code that explicitly 
### mentions it (Code 4: "Treated and admitted for hospitalization (within same facility)")
### submersion has the highest rate

# let's make flag for the disposition we are interested in
neiss_all3['hosp_flag'] = np.where(neiss_all3['Disposition']=='Treated and admitted for hospitalization (within same facility)', 1, 0)

print("Hospitalization rates by diagnosis:")
test = neiss_all3[['Diagnosis','hosp_flag']].groupby('Diagnosis').agg([np.sum, np.size])
test['rate'] = test[('hosp_flag', 'sum')]/test[('hosp_flag', 'size')]
print(test.sort_values(['rate']))

##############################
# What diagnosis most often concluded with the individual leaving without being seen?
##############################

### ANSWER:
### Poisoning has the highest rate

# let's make flag for the disposition we are interested in
neiss_all3['hosp_flag'] = np.where(neiss_all3['Disposition']=='Left without being seen/Left against medical advice', 1, 0)

print("Rates of patient leaving without being seen by diagnosis:")
test = neiss_all3[['Diagnosis','hosp_flag']].groupby('Diagnosis').agg([np.sum, np.size])
test['rate'] = test[('hosp_flag', 'sum')]/test[('hosp_flag', 'size')]
print(test.sort_values(['rate']))

##############################
# Briefly discuss your findings and any caveats you'd mention when discussing this data
##############################

### ANSWER:
# For external clients, I would want to define terminology with them
# I.e. if they wanted to refer to body parts as currenty coded or
# if new groupings would be needed (e.g. if they want to treat 'arm' as a whole
# instead of the current coding split between 'lower arm' and 'upper arm')
# Also note that the data is missing almost 400 body part records, so body part analyses are on partial data

print(len(neiss_all3[(neiss_all3['BodyPart'] == 'Not Recorded')]), 'injuries did not record a body part and', len(neiss_all3[(neiss_all3['Disposition'] == 'Not Recorded')]), 'injuries did not record a disposition')

##############################
# Visualize any existing relationship between age and reported injuries
##############################

age_group = neiss_all3.groupby('age2').size()
age_group.plot.bar(title='Reported injury count by age')

