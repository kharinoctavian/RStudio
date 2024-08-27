writeLines(indramayu_tweet$text)
library(rebus.base)
library(stringr)
#menghapus unsername
str_view_all(indramayu_tweet$text, pattern="@\\w*: ")
indramayu_tweet$stripped_text <- str_replace_all(indramayu_tweet$text, pattern=or("@\\w*: ", "@\\w* "), replacement = "")
#menghapus RT
str_view_all(indramayu_tweet$stripped_text, pattern="RT")
indramayu_tweet$stripped_text<- str_replace_all(indramayu_tweet$stripped_text, pattern="RT ", replacement="")
#menghapus kata kunci pencarian
str_view_all(indramayu_tweet$stripped_text, pattern=or("Indramayu","indramayu","#Indramayu","#indramayu"))
#menghilangkan kata indramayu
indramayu_tweet$stripped_text<-str_replace_all(indramayu_tweet$stripped_text, pattern=or("Indramayu","indramayu","#Indramayu","#indramayu"), replacement="")
#huruf kecil
indramayu_tweet$stripped_text<-str_to_lower(indramayu_tweet$stripped_text)
#menghapus link
str_view_all(indramayu_tweet$stripped_text, pattern=or("http:.*","https:.*"))
indramayu_tweet$stripped_text<- str_replace_all(indramayu_tweet$stripped_text,pattern=or("http:.*","https:.*"), replacement="")
#menghilangkan tanda baca (special character)
indramayu_tweet$stripped_text<-str_replace_all(indramayu_tweet$stripped_text, pattern="[^[:alnum:][:space:]#]", replace="")
#menghapus ð 
indramayu_tweet$stripped_text<- str_replace_all(indramayu_tweet$stripped_text, pattern = "ð", replacement = "")
#menghapus angka
str_view_all(indramayu_tweet$stripped_text, pattern="\\d+")
indramayu_tweet$stripped_text<-str_replace_all(indramayu_tweet$stripped_text,pattern="\\d+", replacement="")
#merapikan double spasi
indramayu_tweet$stripped_text<-str_squish(indramayu_tweet$stripped_text)

library(tokenizers)
library(dplyr)
library(tidytext)
library(wordcloud2)

#buat wordcloud
indramayu_tweet <- indramayu_tweet %>%
  unnest_tokens(output = "word",
                input = stripped_text,
                token = "words", drop = FALSE)
indramayu_tweet$word
tdm <- indramayu_tweet %>%
  count(word, sort=T)
tdm
wordcloud2(tdm)

library(devtools)
library(textclean)
library(katadasaR)

#menghapus slang
spell.lex <- read.csv(file = "D:/colloquial indonesian lexicon.csv", sep = ",")
indramayu_tweet$stripped_text <- replace_internet_slang(indramayu_tweet$stripped_text, slang = paste0("\\b", spell.lex$slang, "\\b"), replacement = spell.lex$formal)

# stemming
stemming <- function(x){
  paste(lapply(x, katadasaR), collapse = " ")
}

stemming_text <- lapply(tokenize_words(indramayu_tweet$stripped_text[]), stemming)
stemming_text[1]

stopword_ind_eng <- readLines("stop_words_ind_eng.txt")
stop_text <- as.character(stemming_text)
stop_text <- tokenize_words(stemming_text, stopwords = stopword_ind_eng)
clean_tweet <- as.character(stop_text)

tweet_token <- indramayu %>%
  unnest_tokens(output = "word",
                token = "words",
                input = stripped_text,
                drop = FALSE)
tdm_old <- tweet_token %>%
  count(word, sort=T)
tdm_old

library(wordcloud)

word_clean <- NULL
for(i in 1:5000){
  word_clean <- c(word_clean, stop_text[[i]])
}
tbl_word_clean <- tibble(word_clean)
tdm <- tbl_word_clean %>%
  count(word_clean, sort=T)
wordcloud2(tdm)
