# Load library, load data, split data
library(text2vec)
library(data.table)
library(magrittr)
data("movie_review")
setDT(movie_review)
setkey(movie_review, id)
set.seed(2017L)
all_ids <- movie_review$id
train_ids <- sample(all_ids, 4000)
test_ids <- setdiff(all_ids, train_ids)
train <- movie_review[J(train_ids)]
test <- movie_review[J(test_ids)]

# Vectorization
prep_fun <- tolower # merubah kedalam lower case 
tok_fun <- word_tokenizer # fungsi tokenizer
it_train <- itoken(train$review,
                   preprocessor = prep_fun,
                   tokenizer = tok_fun,
                   ids = train$id,
                   progressbar = T)
vocab <- create_vocabulary(it_train) # document term matrix

train_tokens <- tok_fun(prep_fun(train$review))

it_train <- itoken(train_tokens,
                   ids = train$id,
                   progressbar = T)
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab) # Fungsi vectorizer

dtm_train <- create_dtm(it_train, vectorizer) # membuat dtm
dim(dtm_train)
identical(rownames(dtm_train), train$id)
dtm_train@x

# Modeling
library(glmnet)
NFOLDS = 4
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
plot(glmnet_classifier)
print(paste('max_auc=', round(max(glmnet_classifier$cvm), 4)))

# Kita sudah punya model 1 = glmnet_classifier
# Test the model
# Note that most text2vec functions are pipe friendly!
it_test = tok_fun(prep_fun(test$review))
# turn off progressbar because it won't look nice in rmd
it_test = itoken(it_test, ids = test$id, progressbar = T)
dtm_test = create_dtm(it_test, vectorizer) # Pembuatan dtm dengan vectorizer dari data training
dtm_test@Dimnames[[2]]
dtm_train@Dimnames[[2]]
preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$sentiment, preds)

# Remove stop_words
stop_words = c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours")
vocab = create_vocabulary(it_train, stopwords = stop_words)
pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
vectorizer = vocab_vectorizer(pruned_vocab)
# create dtm_train with new pruned vocabulary vectorizer

dtm_train  = create_dtm(it_train, vectorizer)
dim(dtm_train)
vocab = create_vocabulary(it_train, ngram = c(1L, 2L))
vocab = prune_vocabulary(vocab, term_count_min = 5,
                         doc_proportion_max = 0.5)

bigram_vectorizer = vocab_vectorizer(vocab)

dtm_train = create_dtm(it_train, bigram_vectorizer)

glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)

dtm_test = create_dtm(it_test, bigram_vectorizer)
preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$sentiment, preds)

# Feature Hashing
h_vectorizer = hash_vectorizer(hash_size = 2 ^ 14, ngram = c(1L, 2L))
t1 = Sys.time()
dtm_train = create_dtm(it_train, h_vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 5,
                              thresh = 1e-3,
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
dtm_test = create_dtm(it_test, h_vectorizer)

preds = predict(glmnet_classifier, dtm_test , type = 'response')[, 1]
glmnet:::auc(test$sentiment, preds)

# Normalization
dtm_train_l1_norm = normalize(dtm_train, "l1")

# TFIDF
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

# define tfidf model
tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf = create_dtm(it_test, vectorizer)
dtm_test_tfidf = transform(dtm_test_tfidf, tfidf)

t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train[['sentiment']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
preds = predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[,1]
glmnet:::auc(test$sentiment, preds)

#####################################
# Load library, data, and split data
library(tidyverse)
library(tidymodels)
library(tidytext)

set.seed(1234)
theme_set(theme_minimal())

# get USCongress data
library(devtools)
library(tibble)
data(USCongress, package = "rcfss")

# topic labels
major_topics <- tibble(
  major = c(1:10, 12:21, 99),
  label = c(
    "Macroeconomics", "Civil rights, minority issues, civil liberties",
    "Health", "Agriculture", "Labor and employment", "Education", "Environment",
    "Energy", "Immigration", "Transportation", "Law, crime, family issues",
    "Social welfare", "Community development and housing issues",
    "Banking, finance, and domestic commerce", "Defense",
    "Space, technology, and communications", "Foreign trade",
    "International affairs and foreign aid", "Government operations",
    "Public lands and water management", "Other, miscellaneous"
  )
)

(congress <- as_tibble(USCongress) %>%
    mutate(text = as.character(text)) %>%
    left_join(major_topics))
set.seed(123)

congress <- congress %>%
  mutate(major = factor(x = major, levels = major, labels = label))
library(rsample)
congress_split <- initial_split(data = congress, strata = major, prop = .8)
congress_split
congress_train <- training(congress_split)
congress_test <- testing(congress_split)

# Preprocessing
library(recipes)
congress_rec <- recipe(major~ text, data=congress_train)
library(textrecipes)
congress_rec <- congress_rec %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = 500) %>%
  step_tfidf(text)

# Train model
library(discrim)
nb_spec <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_spec
library(workflows)

nb_wf <- workflows::workflow() %>%
  add_recipe(congress_rec) %>%
  add_model(nb_spec)
nb_wf
library(stopwords)
library(naivebayes)
nb_wf %>%
  fit(data = congress_train)

# Evaluation
set.seed(123)

nb_wf <- workflow() %>%
  add_recipe(congress_rec) %>%
  add_model(nb_spec)
nb_wf
congress_folds <- vfold_cv(data = congress_train, strata = major)
congress_folds
library(resample)
library(tune)
nb_cv <- nb_wf %>%
  fit_resamples(
    congress_folds,
    control = control_resamples(save_pred = T)
  )
nb_cv_metrics <- collect_metrics(nb_cv)
nb_cv_predictions <- collect_predictions(nb_cv)

nb_cv_metrics
nb_cv_predictions %>%
  group_by(id) %>%
  roc_curve(truth = major, c(starts_with(".pred"), -.pred_class)) %>%
  autoplot() +
  ggplot2::labs(
    color = NULL,
    title = "Receiver operator curve for Congressional bills",
    subtitle = "Each resample fold is shown in a different color"
  )

nb_cv_predictions %>%
  filter(id == "Fold01") %>%
  conf_mat(major, .pred_class) %>%
  autoplot(type = "heatmap") +
  ggplot2::scale_y_discrete(labels = function(x) stringr::str_wrap(x, 20)) +
  ggplot2::scale_x_discrete(labels = function(x) stringr::str_wrap(x, 20))
?str_wrap()

null_classification <- null_model() %>%
  set_engine("parsnip") %>%
  set_mode("classification")

null_cv <- workflow() %>%
  add_recipe(congress_rec) %>%
  add_model(null_classification) %>%
  fit_resamples(
    congress_folds
  )

null_cv %>%
  collect_metrics()
library(ggplot2)
library(forcats)
ggplot(data = congress, mapping = aes(x = fct_infreq(major) %>% fct_rev())) +
  geom_bar() +
  coord_flip() +
  labs(
    title = "Distribution of legislation",
    subtitle = "By major policy topic",
    x = NULL,
    y = "Number of bills"
  )

library(themis)

# build on existing recipe
congress_rec <- congress_rec %>%
  step_downsample(major)
congress_rec
svm_spec <- svm_rbf() %>%
  set_mode("classification") %>%
  set_engine("liquidSVM")

svm_spec
svm_wf <- workflow() %>%
  add_recipe(congress_rec) %>%
  add_model(svm_spec)

svm_wf
set.seed(123)
devtools::install_github("liquidSVM/liquidSVM")
library(e1071)
svm_cv <- fit_resamples(
  svm_wf,
  congress_folds,
  metrics = metric_set(accuracy),
  control = control_resamples(save_pred = TRUE)
)

######################################################
library(readr)
library(dplyr)
library(splitstackshape)
#Text mining packages
library(tm)
library(SnowballC)

getwd()
#loading the data
t1 <- read.csv("ml_text_data.csv")
glimpse(t1)
t1 <- stratified(t1, c("Recommended IND"), 500)
t1 <- t1[,c(2, 5, 7)]
colnames(t1) <- c("Clothing_ID", "Review_Text", "Recommended_IND")
t1[1,]
# Preparing
corpus = Corpus(VectorSource(t1$Review_Text))

corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, tolower)
corpus[[1]][1]

corpus = tm_map(corpus, removePunctuation)
corpus[[1]][1]
corpus = tm_map(corpus, removeWords, c("cloth", stopwords("english")))
corpus[[1]][1]  

corpus = tm_map(corpus, stemDocument)
corpus[[1]][1]  

frequencies = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(frequencies, 0.995)
tSparse = as.data.frame(as.matrix(sparse))
colnames(tSparse) = make.names(colnames(tSparse))
tSparse$recommended_id = t1$Recommended_IND
prop.table(table(tSparse$recommended_id)) #73.6% is the baseline accuracy

#similarity hampir sama kaya korelasi, tp similarity nilainya 0 sampe 1
#kalo pearson korelasi itu -1 sampe 1, krn pembilangnya bukan bentuk kuadratik jd bs negatif
library(text2vec)
sim2(as.matrix(tSparse[1,]), as.matrix(tSparse[2,]), method = "cosine") #yg diharapkan sebesar mungkin
dist(tSparse[1:2,]) #yg diharapkan sekecil mungkin, krn jarak

library(NaiveBayes)


library(caTools)
set.seed(100)
split = sample.split(tSparse$recommended_id, SplitRatio = 0.7)
trainSparse = tSparse[1:800,]
testSparse = tSparse[801:1000,]

library(randomForest)
set.seed(100)
trainSparse$recommended_id = as.factor(trainSparse$recommended_id)
testSparse$recommended_id = as.factor(testSparse$recommended_id )

#Lines 5 to 7
RF_model = randomForest(recommended_id ~ ., data=trainSparse) # Memodelkan dengan data training
predictRF = predict(RF_model, newdata=testSparse)
table(testSparse$recommended_id, predictRF)

# Accuracy
(125+107)/(125+25+43+107) #78%

########################################################
library(keras)
library(dplyr)
library(ggplot2)
library(purrr)
library(pins)
board_register_kaggle(token = "~/kaggle.json")
paths <- pins::pin_get("nltkdata/movie-review", "kaggle")
path <- paths[1]
df <- readr::read_csv("~/Kerja ITS/df_tugas_klasifikasi.csv")
training_id <- sample.int(nrow(df), size = nrow(df)*0.8)
training <- df[training_id,]
testing <- df[-training_id,]
df$text %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary()
num_words <- 10000
max_length <- 50
text_vectorization <- layer_text_vectorization(
  max_tokens = num_words, 
  output_sequence_length = max_length, 
)
text_vectorization %>% 
  adapt(df$text)
get_vocabulary(text_vectorization)
text_vectorization(matrix(df$text[1], ncol = 1))

input <- layer_input(shape = c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(input_dim = num_words + 1, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(input, output)

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

history <- model %>% fit(
  training$text,
  as.numeric(training$tag == "pos"),
  epochs = 10,
  batch_size = 512,
  validation_split = 0.2,
  verbose=2
)
results <- model %>% evaluate(testing$text, as.numeric(testing$tag == "pos"), verbose = 0)
results
