
#Load packages
require(quanteda)
require(quanteda.textmodels)
require(quanteda.textplots)
require(tidyverse)

#Read in data
tweets <-
  #Read in tweets
  read_csv('data/tweets.csv') %>%
  #Join elected officials
  left_join(
    read_csv('data/elected_officials.csv') %>%
      #Pivot so each individual account is a row
      pivot_longer(
        cols = c('officialTwitter',
                 'campaignTwitter',
                 'othertwitter'),
        names_to = 'account_type',
        values_to = 'twitter',
        values_drop_na = TRUE
      ) %>%
      #convert to lowercase so the join works
      mutate('twitter' = tolower(twitter)),
      by = c('username' = 'twitter')) %>%
  #Drop unused columns
  select(State:account_type,
         language,
         username,
         tweet) %>%
  #Filter just to english tweets
  filter(language == 'en') %>%
  #Concatenate tweets by individual name
  group_by(Name) %>% 
  mutate('tweets' = paste0(tweet, collapse = "")) %>%
  #Drop unneeded columns again
  select(!c(tweet,account_type, username)) %>%
  #Select unique rows
  distinct()


# Governor ------------------------------------------------------------

#Running the whole dataset is too big, so filter by office
governors <-
  tweets %>%
  filter(office == 'Governor')

#Convert tweets into tokens
tokens_tweets <- tokens(governors$tweets, 
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_numbers = TRUE,
                        remove_url = TRUE)

#Convert tokens into DTM
dfmat_tweets <- dfm(tokens_tweets,
                    tolower = TRUE)
#Specify 2 rows where index[1] is to the left of index[2]
#Ex. 37 = OR Gov Kate Brown, 1 = AL Gov Kay Ivey
#Fit wordfish to dtm
#NOTE: this line may need to be run more than once
tmod_wf <- textmodel_wordfish(dfmat_tweets, 
                              dir = c(37, 1),
                              sparse = TRUE)
#View summary of ideal points
summary(tmod_wf)

#Add ideal points to governor DF
governors$ideal_point <-
  tmod_wf$theta

#Drop unneeded columns
governors <-
  governors %>%
  select(!c(tweets, language))

#Write to csv
#write_csv(governors, 'data/govs_ideal.csv')

#Same procedure in other sections

# Lt. Governor --------------------------------------------------------

#NOTE: MODEL DID NOT CONVERGE

lt_govs <-
  tweets %>%
  filter(office == 'LtGov')

tokens_tweets <- tokens(lt_govs$tweets, 
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_numbers = TRUE,
                        remove_url = TRUE)
dfmat_tweets <- dfm(tokens_tweets,
                    tolower = TRUE)
tmod_wf <- textmodel_wordfish(dfmat_tweets, 
                              dir = c(8, 43),
                              sparse = TRUE)
summary(tmod_wf)

# governors$ideal_point <-
#   tmod_wf$theta
# 
# governors <-
#   governors %>%
#   select(!c(tweets, language))
# 
# write_csv(governors, 'data/govs_ideal.csv')

# Sec. State ----------------------------------------------------------

#NOTE: Model did not converge

secstate <-
  tweets %>%
  filter(office == 'SecState')

tokens_tweets <- tokens(secstate$tweets, 
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_numbers = TRUE,
                        remove_url = TRUE)
dfmat_tweets <- dfm(tokens_tweets,
                    tolower = TRUE)
tmod_wf <- textmodel_wordfish(dfmat_tweets, 
                              dir = c(17, 7),
                              sparse = TRUE)

# summary(tmod_wf)
# 
# governors$ideal_point <-
#   tmod_wf$theta
# 
# governors <-
#   governors %>%
#   select(!c(tweets, language))
# 
# write_csv(governors, 'data/govs_ideal.csv')
# 

# State AG ------------------------------------------------------------
#NOTE: Model did not converge

stateag <-
  tweets %>%
  filter(office == 'StateAG')

tokens_tweets <- tokens(stateag$tweets, 
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_numbers = TRUE,
                        remove_url = TRUE)
dfmat_tweets <- dfm(tokens_tweets,
                    tolower = TRUE)
tmod_wf <- textmodel_wordfish(dfmat_tweets, 
                              dir = c(1, 21),
                              sparse = TRUE)



# Treasurer -----------------------------------------------------------

treasurer <-
  tweets %>%
  filter(office == 'Treasurer')
  

tokens_tweets <- tokens(treasurer$tweets, 
                        remove_punct = TRUE,
                        remove_symbols = TRUE,
                        remove_numbers = TRUE,
                        remove_url = TRUE)
dfmat_tweets <- dfm(tokens_tweets,
                    tolower = TRUE)
tmod_wf <- textmodel_wordfish(dfmat_tweets, 
                              dir = c(3, 1),
                              sparse = TRUE)

treasurer <-
  treasurer %>%
  select(!c(language, tweets))

treasurer$ideal_point <-
  tmod_wf$theta

write_csv(treasurer, 'data/treasurer_ideal.csv')
