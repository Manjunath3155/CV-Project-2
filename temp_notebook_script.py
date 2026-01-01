#!/usr/bin/env python
# coding: utf-8

# # Recipe Recommender based on NLP

# **Recommender Systems** are employed everywhere in most of the apps that we use. It automatically suggests us the relevant options, topics, music, food, movies to help us make better choices. When it comes to implementing NLP in recommender, it utilizes the similarity of the content. In this project we are using NLP content based filtering to suggest users with top recipes relevant to what data they enter - be it recipe name, ingredient list.  
# 
# In our project we only have text data of recipes, we don't have metadata like cusine type, difficulty level, cook-time, seasonal data or user-generated content (e.g., comments, votes and reviews).

# In[1]:


# Importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string

import nltk
from wordcloud import WordCloud
from scipy.sparse import csr_matrix

from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim import similarities

from gensim.models.coherencemodel import CoherenceModel

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


# ### Loading Data

# In[2]:


file_path1 = "recipes_raw_nosource_ar.json"
allrecipes = pd.read_json(file_path1)
file_path2 = "recipes_raw_nosource_epi.json"
epicurious = pd.read_json(file_path2)
file_path3 = "recipes_raw_nosource_fn.json"
foodnetwork = pd.read_json(file_path3)


# In[3]:


# reding from first file
allrecipes.head()


# In[4]:


# reding from second file
epicurious.head()


# In[5]:


# reding from third file
foodnetwork.head()


# In[6]:


#transposing each dataframe so that index becomes columns
# we can ge a df of shape 39802X4
allrecipes=allrecipes.transpose()
epicurious=epicurious.transpose()
foodnetwork=foodnetwork.transpose()
#merging all the above three dataframes together across rows
#stacking them on top of the other
df=pd.concat([allrecipes, epicurious, foodnetwork])
df.head()


# ## Data Preprocessing

# In[7]:


# checking high level info
df.info()


# We don't need column picture_link for our present scope of work. Also we can drop index and reset it to be numeric.

# In[8]:


# resetting index and dropping unnecessary columns
df=df.reset_index()
df.drop(columns=['index', 'picture_link'], inplace=True)
df.head()


# In[9]:


#checking the null values
df.isna().sum()


# In[10]:


# visually inspecting rows where title is null
df[df['title'].isna()]


# In[11]:


# visually inspecting rows where instructions is null
df[df['instructions'].isna()]


# In[12]:


# visually inspecting rows where ingredients is null
df[df['ingredients'].isna()]


# It appears that most of the rows have all values missing, there are a few exception where there is an empty list or None values in either of the column. There is no rule or pattern observed to try for imputation of these values so we will go ahead and drop all these rows.

# In[13]:


#dropping all the null values
df.dropna(inplace=True)


# In[14]:


df.isna().sum()


# Now we don't have any null values.

# In[15]:


#taking a look at random 20 rows
df.sample(20)


# We see that ingredients column is a list of items. Just to avoid a situation where we have a same ingredient repeated we will convert the list to a set.
# 
# Also, we will claculate the number of items in each set, which will be equal to number of unique items in the ingredients.

# In[16]:


#lets convert the list of items to set to avoid any repeated ingredients
df['ingredients']=df['ingredients'].apply(set)

#Number of elements in ingredients list
df['ingredient_item']=df['ingredients'].str.len()


# In[17]:


#changing the display settings to see complete text
pd.set_option('display.max_colwidth', None)


# In[18]:


df['ingredient_item'].max()


# In[19]:


#recipe with maximum number of ingredients
df.loc[df['ingredient_item']==df['ingredient_item'].max()]


# In[20]:


print(df['ingredients'][4])


# There is a word ADVERTISEMENT that appears in most of the ingredient column items and we need to remove it.
# 
# We can also see the newline \n in the instructions column which works fine in the print statements.

# In[21]:


print(df['instructions'][20])


# **Idea to add new feature**
# 
# Can we add a new column that specifies the time to cook?
# - Do all the instructions have time mentioned in the description?
# - Can we sum that time and say it is the total time to cook the recipe?

# In[22]:


df.sample(2)['instructions']


# Only some of the instructions have mention about cook and wait time but not all the instructions. Also, there is mention of ingredient quantities in number which might get parsed when extracting numbers from instructions. So we will not focus on getting this information out of the instructions as it may result in null or incorrect values.
# 

# In[23]:


# adding ; in place if comma in list of items
df['items'] = df['ingredients'].apply(lambda x: ';'.join(map(str, x)))


# In[24]:


df.head(2)


# In[25]:


#replacing the word ADVERTISEMENT
df['items']=df['items'].replace('ADVERTISEMENT', '', regex=True)
df.head()


# In[26]:


#finding out recipes where ingredient requirement is minimum
df.loc[df['ingredient_item']==df['ingredient_item'].min()]


# - Interesting to find out that 1520 rows have no ingredient items mentioned in the data set while we have complete recipes.
# - One more thing, in row entries 125146, 125115 there is a mention of pic credit. We need to find out if there are many instruction where we have this thing mentioned. If it is so we need to take care of it.

# In[27]:


df.loc[df['ingredient_item']==df['ingredient_item'].min()].index


# We will drop all these rows where the ingredient list is empty.

# In[28]:


df.drop(labels=df.loc[df['ingredient_item']==df['ingredient_item'].min()].index, axis=0, inplace=True )


# We can check one more thing - if the recipe instructions is too small to be considered a recipe.

# In[29]:


df['recipe_words']=df['instructions'].apply(lambda x: len(x.split()))
df.loc[df['recipe_words']==df['recipe_words'].min()]


# In[30]:


df.loc[df['recipe_words']==df['recipe_words'].min()].shape


# So we can see that these 15 rows don't have any instructions, we can drop these rows from our data.

# In[31]:


df.drop(labels=df.loc[df['recipe_words']==df['recipe_words'].min()].index, axis=0, inplace=True )


# In[32]:


df.loc[df['recipe_words']==df['recipe_words'].min()].shape


# In[33]:


df.loc[df['recipe_words']==df['recipe_words'].min()]


# There are 20 more recipes where the instructions are not there. It only had a full stop or only a random text which was recognized as one word. We should delete it further.

# In[34]:


df.drop(labels=df.loc[df['recipe_words']==df['recipe_words'].min()].index, axis=0, inplace=True )


# We need to check one more thing.
# - Which recipes have mention of 'Photgraph by'.

# In[35]:


df[df['instructions'].str.contains('Photograph')]


# In[36]:


df_clean=df.copy()
#find rows which contain 'Photograph by XXX XXX'
mask=df[df['instructions'].str.contains('Photograph')]

# remove the last four words from the instructions which contain 'Photograph by XXX XXX'
df_clean.loc[mask.index, 'instructions']=mask['instructions'].map(lambda x: ' '.join(x.split()[:-4]))
df_clean.head()


# Yey! Looks like we now have a cleaned data set to work with.

# In[37]:


df_clean.iloc[:,[0, 2, 3, 4, 5]].duplicated().sum()


# Ohh! we have 1085 rows with duplicated values.
# 
# Final visual checks before we drop them.

# In[38]:


df_clean.loc[df_clean.iloc[:,[0, 2, 3, 4, 5]].duplicated()]


# In[39]:


df_clean.loc[df['title']=='Green Beans With Walnuts']


# So our query to check duplicates was correct. We will go ahead and drop the duplicates.

# In[40]:


#dropping ingredients column as it doesn't allow to check for duplicates since it is a list and we already
#have the same information in the items column
df_clean.drop(columns='ingredients', inplace=True)
df_clean.drop_duplicates(keep='first', inplace=True)


# In[41]:


df_clean.shape


# Finally, we have the clean dataset with 121833 unique recipes and 5 columns.
# 
# 

# In[42]:


df_clean['items'][599]


# #### Indepth data cleaning
# 
# We will now remove the stop words, white spaces, numbers, punctuations and do lemmetization on the text data so that we are only left with words of our interest. I can also remove measurement units form the text such as miligram, grams, kg, oz as these words will appear very often in the text and may not necessarily give semantic meaning.

# In[43]:


#cell block to clean the text data - removing white spaces, numbers, punctuations, stop-words and does lemmetization

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download('omw-1.4', quiet=True)

# Function for text standardization
def standardize_text(input_text):
    cleaned_text = []

    for text in input_text:
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.replace('\n',' ') # Remove New Lines
        text = text.strip() # Remove Leading White Space
        text = re.sub(' +', ' ', text) # Remove multiple white spaces

        # Tokenize the ingredient
        tokens = word_tokenize(text.lower())

        # Unit normalization mapping
        unit_mapping = {
            "tsp": "teaspoon",
            "tbsp": "tablespoon",
            "oz": "ounce",
            "g": "gram",
            "lb": "pound"
            }

        # Standardize units and remove quantities
        standardized_tokens = []
        for token in tokens:
            # Remove quantities (check for numeric tokens)
            if not token.isnumeric():
                # Normalize units
                if token in unit_mapping:
                    token = unit_mapping[token]
                standardized_tokens.append(token)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        standardized_tokens = [lemmatizer.lemmatize(token) for token in standardized_tokens]

        # Stop word removal
        stop_words = set(stopwords.words("english"))
        standardized_tokens = [token for token in standardized_tokens if token not in stop_words]

        # Words to remove
        common_words= ["teaspoon", "tablespoon", "ounce", "gram", "pound", "cup", "chopped", "fresh", "ground", "large", "sliced", "peeled",
                       "cut", "freshly", "finely", "plus", "white", "clove", "room", "dry" , "inch"]
        standardized_tokens = [token for token in standardized_tokens if token not in common_words]

        # Join the standardized tokens back to form the ingredient string
        standardized_text = " ".join(standardized_tokens)

        cleaned_text.append(standardized_text)

    return cleaned_text


# sliced peeled cut freshly finely + plus + white + red +minced + flour1 + room + melted + dry + grated these extra words can further be added.

# In[44]:


standardized_ingredients=standardize_text(df_clean['items'])


# In[45]:


standardized_ingredients[788]


# ### Exploratory Data Analysis
# #### Token in Recipe Ingredients

# In[46]:


from collections import Counter
# Calculate word frequencies
word_freq = Counter(" ".join(standardized_ingredients).split())
# Sort word frequencies in descending order
sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))


# In[47]:


# Top N most frequent words to display
top_n = 20

# Get the top N words and their frequencies
top_words = list(sorted_word_freq.keys())[:top_n]
top_freq = list(sorted_word_freq.values())[:top_n]

plt.figure(figsize=(10,6))
sns.barplot(x=top_words, y=top_freq, color='seagreen')
plt.xticks(rotation=45)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top {} Most Frequent Ingrediennt Item'.format(top_n))
plt.tight_layout()
plt.show()


# In[48]:


from wordcloud import WordCloud
# Concatenate all the standardized ingredients into a single text
text_data = " ".join(standardized_ingredients)
# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# Display the generated word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Standardized Ingredients')
plt.show()


# #### Recipes with maximum number of ingredients

# In[49]:


#function to wrap xaxix tick names
#Add new line after three words
def wrap_text(sentence):
    a=sentence.split()
    n=2
    ret= ''
    for i in range(0, len(a), n):
         ret += ' '.join(a[i:i+n]) + '\n'
    return ret
df_plot=df_clean[['title', 'ingredient_item']].sort_values(by='ingredient_item', ascending=False).head(10)
df_plot['wrapped_text']=df_plot['title'].apply(lambda x: wrap_text(x))
plt.figure(figsize=(15,6))
sns.barplot(data=df_plot,
            x='title', y='ingredient_item', color='seagreen')
plt.xlabel('Recipe Names')
plt.ylabel('Number of Ingredients')
plt.xticks(range(0,len(df_plot.index)), df_plot['wrapped_text'].values, rotation=30)
plt.title('Top 10 recipes with maximum number of Ingredients')
plt.show()


# #### Combining all text data and standardizig it

# Okay, now I am planning to add all text from `title`, `instructions` and `ingredient_item` together so that the search algorithm to find the recommended recipes could make better search even if user knows just the recipe name or a few of the ingredients.

# In[50]:


all_text=df_clean['title']+ ' '+df_clean['items']+' '+df_clean['instructions']


# In[51]:


cleaned_ver=standardize_text(all_text)


# In[52]:


# df_all.info()
df_all=pd.DataFrame(cleaned_ver)
df_all.info()


# In[53]:


df_all=df_all.rename(columns={0: 'combined_text'})


# ### Recommendation using text similarity
# 
# #### Implementing TFIDF Vectorization and Cosine Similarity

# Let us start with TFIDF vectorization to convert text into numerical representation and then we will implement recommender logic using cosine similarity.
# Later on we can try working with word2vec and topic modelling to experiment and learn which approach gives the most relevant recipes.

# In[54]:


df_all.info()


# In[55]:


df_all.head(2)


# In[56]:


combined_text= df_all['combined_text'].tolist()


# In[57]:


# combined_text=cleaned_ver
combined_text[15238]


# In[58]:


combined_text[706]


# There are some tokens like 13X9, 12X10inch, 1/2, 160deg F etc which are not removed so we can try two approchaes to get rid of them.
# 1. We can adjust min_df=5 to see if these tokens are removed during TFIDF vectorization
# 2. We can remove these by using regex matching.
# 
# 

# In[59]:


from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(lowercase= True, min_df=5)

# Apply TF-IDF vectorization to the combined text
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)


# In[60]:


tfidf_words = tfidf_vectorizer.get_feature_names_out()
print(tfidf_matrix.shape)
print(len(tfidf_words))


# In[61]:


tfidf_words[100:800]


# Okay, so it seems that we were not able to remove this pattern completely. Also, there are some words with number preceeding measurement like - 4pound', '4pounds', '4quart', '4rib', '4serving',
#        '4sided', '4th', '4to', '4to5pound', '4x12'

# In[62]:


def find_items_with_patterns(item_list, pattern):
    matching_items = []
    for item in item_list:
        if re.search(pattern, item):
            matching_items.append(item)
    return matching_items

# Regular expression pattern to match patterns like "13by30inch"
pattern = r'\d+(?:\/\d+)?(?:[a-z°°]+)?'

# Find items with numbers or patterns
matching_items = find_items_with_patterns(combined_text, pattern)


# In[63]:


matching_items[99]


# In[64]:


def custom_tokenizer(text):
    tokens = text.split()
    tokens= [re.sub(r'\d+$', '', token) for token in tokens]
    # Filter out measurement patterns
    tokens= [token for token in tokens if not re.match(r'^\d+[A-Za-z]+\d+$', token)]
    filtered_tokens = [token for token in tokens if not re.match(r'\d+(?:\/\d+)?(?:[a-z°°]+)?', token)]

    return filtered_tokens


from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase= True, max_features=50000)

# Apply TF-IDF vectorization to the combined text
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)


# In[65]:


tfidf_words = tfidf_vectorizer.get_feature_names_out()
print(tfidf_matrix.shape)
print(len(tfidf_words))


# In[66]:


tfidf_words[100:900]


# #### User input vectorization using TF-IDF

# In[67]:


# Example user input (replace this with actual user input after preprocessing)
user_input = "ground beef, pasta, spaghetti, tomato sauce, bacon, onion, zucchini, and, cheese"


# #### Finding recipes using Cosine Similarity

# In[68]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Function to calculate cosine similarity between a user input vector and recipe vectors
def calculate_similarity(user_input_vector, recipe_matrix):
    return cosine_similarity(user_input_vector, recipe_matrix)

def find_recipes(user_input):

    # Generate the user input vector using the TFIDF model
    user_input_vector = tfidf_vectorizer.transform([user_input])

    # Calculate similarity between user input vector and recipe vectors
    similarities = calculate_similarity(user_input_vector, tfidf_matrix )
    return similarities

similarities=find_recipes(user_input)


# In[69]:


# np.argsort(similarities)


# In[70]:


# np.where(df_clean.index== 79928)


# In[71]:


#resetting index from df_clean since we dropped some rows while preprocessing
df_clean = df_clean.reset_index(drop=True)


# In[72]:


# Sort the recipes based on similarity and get top N recommendations
top_n = 5
top_indices = np.argsort(similarities[0])[::-1][:top_n]
top_recipes = [df_clean['title'][i] for i in top_indices]

print(f"Top {top_n} Recipe Recommendations:")
for i, recipe in enumerate(top_recipes, 1):
    print(f"{i}. {recipe}")


# In[73]:


#printing out 
df_clean.loc[top_indices, ['title', 'items', 'instructions']]


# #### Implementing Word2Vec Embedding

# Since we observed that some of the patterns where not removed even after standardization and therefore we will remove these patterns before we implement word2Vec.
# 
# Later we will build the word2vec model on our recipe corpora and use it to transform user input data. Later using cosine similarity will try to find the top five recommended recipes.

# In[74]:


# Let us remove the words/tokens from our text before feeding it to word2vec model.
def remove_patterns(list_text):
    text_wo_pattern_lst=[]
    for text in list_text:
        tokens = text.split()
        tokens= [re.sub(r'\d+$', '', token) for token in tokens]
        # Filter out measurement patterns
        tokens= [token for token in tokens if not re.match(r'^\d+[A-Za-z]+\d+$', token)]
        filtered_tokens = [token for token in tokens if not re.match(r'\d+(?:\/\d+)?(?:[a-z°°]+)?', token)]

        filtered_text=" ".join(filtered_tokens)
        text_wo_pattern_lst.append(filtered_text)

    return text_wo_pattern_lst


# In[75]:


clean_text_lst=remove_patterns(combined_text)


# In[76]:


len(clean_text_lst)


# In[77]:


import gensim
from gensim.models import Word2Vec

# all_tokens=word_tokenize(cleaned_ver)
all_tokens=[]
for i in clean_text_lst:
    all_tokens.append(word_tokenize(i))

# Train Word2Vec model
w2v_model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, workers=4)


# In[78]:


w2v_model.wv['fillet']


# In[79]:


len(all_tokens)


# In[80]:


w2v_model.wv.most_similar('beef')


# In[81]:


len(w2v_model.wv.index_to_key)


# In[82]:


# Generate user input vector
user_input = "ground beef, pasta, spaghetti, tomato sauce, bacon, onion, zucchini, and, cheese"
# user_input = standardize_text(user_input)
user_input_tokens = word_tokenize(user_input)

# Initialize an empty user input vector
user_input_vector = [0] * w2v_model.vector_size

# Calculate the mean vector of the user input tokens
num_tokens = 0
for token in user_input_tokens:
    if token in w2v_model.wv:
        user_input_vector = [a + b for a, b in zip(user_input_vector, w2v_model.wv[token])]
        num_tokens += 1

if num_tokens > 0:
    user_input_vector = [x / num_tokens for x in user_input_vector]



# In[83]:


recipe_vector=[0]*w2v_model.vector_size
all_recipes_vector=[]
for i in all_tokens:
   # Calculate the mean vector of the each recipe
    num_tokens = 0 
    for token in i:
        if token in w2v_model.wv:
            recipe_vector = [a + b for a, b in zip(recipe_vector, w2v_model.wv[token])]
            num_tokens += 1

    if num_tokens > 0:
        recipe_vector = [x / num_tokens for x in recipe_vector]
    all_recipes_vector.append(recipe_vector)


# In[84]:


all_recipes_vector=np.array(all_recipes_vector)


# In[85]:


user_input_vector = np.array(user_input_vector).reshape(1, -1)
# Calculate cosine similarity between user input and all recipes
similarities2 = cosine_similarity(user_input_vector, all_recipes_vector)

# Recommend top recipes
# Get the indices of top N similar recipes
N = 5  # Number of recipes to recommend
top_indices2 = similarities2.argsort()[0][-N:][::-1]

# Print the top N recommended recipes
for idx in top_indices2:
    print(df_clean['title'][idx])


# In[86]:


df_clean.loc[top_indices2, ['title', 'items', 'instructions']]


# ## Recommendation based on Topic Modeling

# ### Topic modeling using Gensim based LDA on BOW

# Iniotially, I am going ahead with modelling arbitary 10 topics and later will use grid search to find the optimal number of topics.

# In[87]:


from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim import similarities

# Step 1: Prepare the Recipe Data


tokenized_recipes=[]
for i in clean_text_lst:
    tokenized_recipes.append(word_tokenize(i))

# Step 2: Create a Gensim Dictionary
dictionary = corpora.Dictionary(tokenized_recipes)

# Step 3: Create a Gensim Corpus
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_recipes]

# Step 4: Train the LDA Model

num_topics = 10  # Specify the number of topics
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)



# In[88]:


# Step 5: Transform User Input
user_input = "ground beef, pasta, spaghetti, tomato sauce, bacon, onion, zucchini, cheese"
user_input_tokens = [token for token in standardize_text(user_input)]
user_input_bow = dictionary.doc2bow(user_input_tokens)

# Step 6: Get Topic Distribution for User Input
user_topic_distribution = lda_model[user_input_bow]

# Step 7: Recommend Recipes
# Calculate similarity scores between user_topic_distribution and recipe topic distributions
index = similarities.MatrixSimilarity(lda_model[corpus])
sims = index[user_topic_distribution]

# Sort recipes by similarity scores
sorted_recipes = sorted(enumerate(sims), key=lambda item: -item[1])

# Get the top recommended recipes
top_recipes = sorted_recipes[:5]

# Print the top recommended recipe indices
print(top_recipes)


# In[89]:


top_recipes_index=[i[0] for i  in top_recipes]
top_recipes_index


# In[90]:


top_recipes = [df_clean['title'][i] for i in top_recipes_index]

print(f"Top {top_n} Recipe Recommendations:")
for i, recipe in enumerate(top_recipes, 1):
    print(f"{i}. {recipe}")


# In[91]:


df_clean.iloc[top_recipes_index]


# In[92]:


lda_model.show_topics(formatted=False)


# #### Using grid search and coherence score to find optimal number of topics

# In[93]:


from gensim.models.coherencemodel import CoherenceModel
def compute_coherence_values_lda_model(dictionary, corpus, texts, limit, start=2, step=6):
    coherence_values_lda = []
    model_list_lda = []
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics,
                                       chunksize=2000,
                                       alpha='auto',
                                       eta='auto',
                                       iterations=400,
                                       passes=20,
                                       eval_every=None)
        print(num_topics)
        model_list_lda.append(lda_model)
        coherencemodel_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values_lda.append(coherencemodel_lda.get_coherence())

    return model_list_lda, coherence_values_lda


# In[ ]:


model_list_lda, coherence_values_lda = compute_coherence_values_lda_model(dictionary=dictionary,corpus=corpus, texts=tokenized_recipes, start=10, limit=50, step=10)


# In[ ]:


model_list_lda, coherence_values_lda


# Lets us visualize and see how the coherence score is varying.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Show graph
limit=50; start=10; step=10;
x = range(start, limit, step)
y_lda = coherence_values_lda

plt.plot(x, y_lda, label='LDA_model')

plt.xlabel("Num Topics")
plt.ylabel("Coherence score")    
plt.legend()
plt.savefig('model_lda_coherence_score.png')
plt.show()


# Clearly, we can see that there is a decreasing trend and score is highest for 10 number of topics.

# In[ ]:


# Print the coherence scores
for m, cv in zip(x, coherence_values_lda):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# The optimal model of LDA using BOW corpus is the one with higher coherence score:

# In[ ]:


optimal_model_lda = model_list_lda[0]

optimal_model_lda.show_topics()


# In[ ]:


def create_wordcloud(model, topic):
    text = {word: value for word, value in model.show_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Topic" + " "+ str(topic))
    plt.show()

 #visualize the first 3 topics for LDA with BOW corpus
for i in range(1,4):
    create_wordcloud(optimal_model_lda, topic=i)


# In[ ]:


# Step 5: Transform User Input
user_input = "ground beef, pasta, spaghetti, tomato sauce, bacon, onion, zucchini, cheese"
user_input_tokens = [token for token in standardize_text(user_input)]
user_input_bow = dictionary.doc2bow(user_input_tokens)

# Step 6: Get Topic Distribution for User Input
user_topic_distribution = optimal_model_lda[user_input_bow]

# Step 7: Recommend Recipes
# Calculate similarity scores between user_topic_distribution and recipe topic distributions
index = similarities.MatrixSimilarity(optimal_model_lda[corpus])
sims = index[user_topic_distribution]

# Sort recipes by similarity scores
sorted_recipes = sorted(enumerate(sims), key=lambda item: -item[1])

# Get the top recommended recipes
top_recipes = sorted_recipes[:5]

# Print the top recommended recipe indices
print(top_recipes)


# In[ ]:


top_recipes = [df_clean['title'][i] for i in top_recipes_index]

print(f"Top {top_n} Recipe Recommendations:")
for i, recipe in enumerate(top_recipes, 1):
    print(f"{i}. {recipe}")


# In[ ]:


df_clean.iloc[top_recipes_index]


# #### Conclusion

# Since we don't have a labelled data our model is unsupervised and hence evaluation materics is not there. However from observing the result and comparing the models we found that only **Word2Vec gave out the relevant recipes.**
# 
# Later I would like to make a streamlit app that users can use and play around.

# #### Playing around with Word2Vec model

# In[ ]:


user_input='chicken, spinach'


# In[ ]:


#vectorize user input as per Word2Vec Model

def user_input_vectorize(word2vec_model, user_input):

    user_input_tokens = word_tokenize(user_input)

    # Initialize an empty user input vector
    user_input_vector = [0] * word2vec_model.vector_size

    # Calculate the mean vector of the user input tokens
    num_tokens = 0
    for token in user_input_tokens:
        if token in word2vec_model.wv:
            user_input_vector = [a + b for a, b in zip(user_input_vector, word2vec_model.wv[token])]
            num_tokens += 1

    if num_tokens > 0:
        user_input_vector = [x / num_tokens for x in user_input_vector]

    return user_input_vector

def recommend_w2v(user_input_vector):
    user_input_vector = np.array(user_input_vector).reshape(1, -1)
    # Calculate cosine similarity between user input and all recipes
    similarities2 = cosine_similarity(user_input_vector, all_recipes_vector)

    # Recommend top recipes

    N = 5  # Number of recipes to recommend
    top_indices2 = similarities2.argsort()[0][-N:][::-1]

    return top_indices2


# In[ ]:


indices=recommend_w2v(user_input_vectorize(w2v_model , user_input))


# In[ ]:


# Print the top N recommended recipes
for idx in indices:
    print(df_clean['title'][idx])


# In[ ]:


df_clean.loc[indices, ['title', 'items', 'instructions']]


# In[ ]:


user_input='zuchinni, carrot, lamb meat, wheat flour, onion'


# In[ ]:


indices=recommend_w2v(user_input_vectorize(w2v_model , user_input))
# Print the top N recommended recipes
for idx in indices:
    print(df_clean['title'][idx])


# In[ ]:


df_clean.loc[indices, ['title', 'items', 'instructions']]


# Given that our current constraints of not having labels or user feedback, we are stopping here. However, there are a few aspects to consider and if possible I would like to implement in future:
# 
# - Evaluation Metrics: We could compare the average cosine scores of the different approaches we employed in our project but I am stopping here since the result of LDA based model appeared very inaccurate. A higher average similarity score generally indicates better recommendations.
# 
# - User Engagement: Monitor user engagement with the recommender system. Track metrics such as the number of clicks on recommended recipes, the time spent on recipe pages, and the number of interactions with the recommendations. These metrics can provide insights into how well your system is performing in the absence of explicit user feedback.
# 
# - Tuning Hyperparameters: Experiment with different hyperparameters in your recommendation models (e.g., number of topics in LDA, TF-IDF vectorization settings) to see if they impact the quality of recommendations. Already tried tuning the number of topics on LDA but tuned on a limited set as it was taking a long run time.
# 
# - User Surveys: Although we don't have explicit feedback, we can conduct user surveys or collect feedback through user interviews to gather qualitative information about their satisfaction with the recommendations.
# 
