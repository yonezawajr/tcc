# Carregar bibliotecas necessárias
library(dplyr)
library(tm)
library(purrr)
library(tidyr)
library(tidytext)
library(readr)
library(stringr)
library(stopwords)
library(stringi)
library(readxl)
library(ggplot2)
library(caret)
library(randomForest)
library(gbm)
library(ngram)
library(NLP)
library(spacyr)
library(pROC)
library(PRROC)

# Funções de pré-processamento
removePunctuation <- function(x) {
  gsub("[[:punct:]]+", "", x)
}
removeNumbers <- function(x) {
  gsub("[[:digit:]]+", "", x)
}
removeAccents <- function(x) {
  stri_trans_general(x, "Latin-ASCII")
}

# Função para ler as planilhas com descrições cirúrgicas
arquivo <- "E:/Projeto MBA/dataset/dataset.xlsx"
sheets <- excel_sheets(arquivo)
tabela_tuss <- read_excel(arquivo, sheet = sheets[1])

ler_planilha <- function(sheet) {
  data <- read_excel(arquivo, sheet = sheet)
  data %>%
    mutate(procedimento = as.character(procedimento),
           descricao = as.character(descricao),
           tuss = as.character(tuss)) %>%
    select(procedimento, descricao, tuss)
}
dataset <- sheets[-1] %>% map_df(ler_planilha)

# Obter stopwords padrão em português
stopwords_pt <- stopwords(language = "pt")

# Pré-processamento das descrições
dataset <- dataset %>%
  filter(str_trim(descricao) != "") %>%
  mutate(descricao = str_squish(descricao),
         descricao = tolower(descricao),
         descricao = removePunctuation(descricao),
         descricao = removeNumbers(descricao),
         descricao = removeAccents(descricao),
         id = row_number())

dataset_tidy <- dataset %>%
  unnest_tokens(word, descricao) %>%
  filter(!word %in% stopwords_pt) %>%
  mutate(word_id = as.integer(factor(word)))

corpus <- Corpus(VectorSource(dataset_tidy$word))
dtm <- DocumentTermMatrix(corpus)
dtm_matrix <- as.matrix(dtm)

# Clusterização usando K-Means
set.seed(123)
clusters <- kmeans(dtm_matrix, centers = 5, nstart = 25)  
dataset_tidy <- dataset_tidy %>%
  mutate(cluster = clusters$cluster[match(id, unique(id))])

# Estatísticas descritivas
word_count_per_descricao <- dataset_tidy %>%
  group_by(id) %>%
  summarise(word_count = n())

max_words_per_descricao <- word_count_per_descricao %>%
  summarise(max_words = max(word_count))

mean_words_per_descricao <- word_count_per_descricao %>%
  summarise(mean_words = mean(word_count))

std_dev_words_per_descricao <- word_count_per_descricao %>%
  summarise(std_dev_words = sd(word_count))

word_count_distribution <- word_count_per_descricao %>%
  count(word_count)

unique_words_count <- dataset_tidy %>%
  summarise(unique_words = n_distinct(word))

procedimento_count <- dataset %>%
  summarise(procedimento = n_distinct(procedimento))

procedures_per_tuss <- dataset %>%
  count(tuss, sort = TRUE)

procedure_frequencies <- dataset %>%
  count(procedimento, sort = TRUE)

cat("Contagem de Palavras Únicas: ", unique_words_count$unique_words, "\n")
cat("Quantidade de Procedimentos: ", procedimento_count$procedimento, "\n")
cat("Tamanho Máximo de Palavras por Descrição: ", max_words_per_descricao$max_words, "\n")
cat("Tamanho Médio de Palavras por Descrição: ", mean_words_per_descricao$mean_words, "\n")
cat("Desvio Padrão do Número de Palavras por Descrição: ", std_dev_words_per_descricao$std_dev_words, "\n")
cat("\nDistribuição do Número de Palavras por Descrição:\n")
print(word_count_distribution)
cat("\nContagem de Procedimentos por Código TUSS:\n")
print(procedures_per_tuss)
cat("\nFrequência dos Procedimentos Mais Comuns:\n")
print(procedure_frequencies)

#Plotagem dos gráficos
word_count_distribution_df <- as.data.frame(word_count_distribution)
ggplot(word_count_distribution_df, aes(x = word_count, y = n)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Distribuição do Número de Palavras por Descrição",
       x = "Número de Palavras por Descrição",
       y = "Frequência") +
  theme_minimal()

procedure_frequencies_sorted <- procedure_frequencies %>%
  arrange(desc(n)) %>%
  head(15)
ggplot(procedure_frequencies_sorted, aes(x = reorder(procedimento, n), y = n)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Top 15 Procedimentos Mais Comuns",
       x = "Procedimento",
       y = "Frequência") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  coord_flip()

stats_df <- data.frame(
  Tipo = c("Tamanho Máximo", "Tamanho Médio", "Desvio Padrão"),
  Valor = c(max_words_per_descricao$max_words, mean_words_per_descricao$mean_words, std_dev_words_per_descricao$std_dev_words)
)
ggplot(stats_df, aes(x = Tipo, y = Valor, fill = Tipo)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  geom_text(aes(label = round(Valor, 2)), vjust = -0.3, size = 4) +
  labs(title = "Estatísticas de Tamanho das Descrições",
       y = "Número de Palavras",
       x = "") +
  theme_minimal()

# TF-IDF
tf <- dataset_tidy %>%
  count(id, word_id) %>%
  group_by(id) %>%
  mutate(tf = n / sum(n)) %>%
  ungroup()

df <- dataset_tidy %>%
  distinct(id, word_id) %>%
  count(word_id) %>%
  mutate(df = n)

total_docs <- length(unique(dataset_tidy$id))
idf <- df %>%
  mutate(idf = log(total_docs / df))
tf_idf <- tf %>%
  left_join(idf, by = "word_id") %>%
  mutate(tf_idf = tf * idf) %>%
  arrange(desc(tf_idf))
tf_idf <- tf_idf %>%
  left_join(dataset %>% select(id, procedimento), by = "id")
print(tf_idf)

# Aplicar NER
spacy_initialize(model = "pt_core_news_sm")
#ner_results <- spacy_parse(dataset$descricao, entity = TRUE)
#ner_results <- spacy_extract_entity(dataset$descricao)
dataset_ner <- dataset %>%
  group_by(id) %>%
  mutate(entities = list(spacy_extract_entity(descricao))) %>%
  ungroup()

# Criar n-gramas
ngram_df <- dataset %>%
  unnest_tokens(ngram, descricao, token = "ngrams", n = 2) %>%
  filter(!ngram %in% stopwords_pt) %>%
  count(id, ngram)

# Preparar os dados para o treinamento
dataset_treinamento <- tf_idf %>%
  select(-id) %>% 
  filter(!is.na(procedimento))
dataset_treinamento$procedimento <- make.names(dataset_treinamento$procedimento)

# Treinamento de Modelos
set.seed(123)
train_index <- createDataPartition(dataset_treinamento$procedimento, p = .8, list = FALSE)
train_data <- dataset_treinamento[train_index, ]
test_data <- dataset_treinamento[-train_index, ]

# Random Forest
tuneGrid_rf <- expand.grid(mtry = seq(1, ncol(train_data) - 1, by = 2))
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE)
set.seed(123)
model_rf <- train(procedimento ~ ., data = train_data, method = "rf",
                  trControl = train_control, tuneGrid = tuneGrid_rf, verbose = TRUE)

# Avaliação dos Modelos
rf_pred <- predict(model_rf, newdata = test_data)
rf_probs <- predict(model_rf, newdata = test_data, type = "prob")
combined_levels <- union(levels(rf_pred), levels(test_data$procedimento))
print(combined_levels)

rf_pred <- factor(rf_pred, levels = combined_levels)
print(levels(rf_pred))
test_data$procedimento <- factor(test_data$procedimento, levels = combined_levels)
print(levels(test_data$procedimento))

rf_cm <- confusionMatrix(rf_pred, test_data$procedimento)
print(rf_cm)

# Curva ROC e Precision-Recall para uma classe escolhida
classe_escolhida <- levels(test_data$procedimento)[1]

rf_roc <- roc(as.numeric(test_data$procedimento == classe_escolhida), rf_probs[, classe_escolhida])
plot(rf_roc, main = "Curva ROC - Random Forest", col = "blue")
legend("bottomright", legend = paste("AUC =", round(auc(rf_roc), 2)), col = "blue", lty = 1, cex = 0.8)

rf_pr <- pr.curve(scores.class0 = rf_probs[, classe_escolhida], weights.class0 = test_data$procedimento == classe_escolhida, curve = TRUE)
plot(rf_pr, main = "Curva Precision-Recall - Random Forest", col = "blue")
legend("bottomright", legend = paste("AUC =", round(rf_pr$auc.integral, 2)), col = "blue", lty = 1, cex = 0.8)

# GBM 
gbm_model <- gbm(procedimento ~ ., 
                 data = train_data, 
                 distribution = "multinomial",
                 n.trees = 100, 
                 interaction.depth = 3, 
                 shrinkage = 0.01, 
                 n.cores = NULL,  
                 train.fraction = 1,  
                 n.minobsinnode = 10, 
                 verbose = TRUE) 

# Avaliação do Modelo GBM
gbm_pred <- predict(gbm_model, newdata = test_data, n.trees = 100, type = "response")
gbm_pred_class <- factor(apply(gbm_pred, 1, function(x) colnames(gbm_pred)[which.max(x)]), levels = levels(test_data$procedimento))

if(length(gbm_pred_class) == length(test_data$procedimento)) {
  gbm_cm <- confusionMatrix(gbm_pred_class, test_data$procedimento)
  print(gbm_cm)
} else {
  print("Os comprimentos de gbm_pred_class e test_data$procedimento não coincidem.")
}

# Calcular as probabilidades das classes para os dados de teste - GBM
gbm_probs <- as.data.frame(gbm_pred)
classe_escolhida <- sub("\\.100$", "", colnames(gbm_probs)[1])
print(paste("Classe escolhida:", classe_escolhida))
print(unique(test_data$procedimento))
test_data$procedimento <- factor(test_data$procedimento)

# Curva ROC e Precision-Recall para a classe escolhida
if (sum(test_data$procedimento == classe_escolhida) > 0 && sum(test_data$procedimento != classe_escolhida) > 0) {
  gbm_roc <- roc(as.numeric(test_data$procedimento == classe_escolhida), gbm_probs[[paste0(classe_escolhida, ".100")]])
  plot(gbm_roc, main = "Curva ROC - GBM", col = "red")
  legend("bottomright", legend = paste("AUC =", round(auc(gbm_roc), 2)), col = "red", lty = 1, cex = 0.8)
  
  gbm_pr <- pr.curve(scores.class0 = gbm_probs[[paste0(classe_escolhida, ".100")]], weights.class0 = test_data$procedimento == classe_escolhida, curve = TRUE)
  plot(gbm_pr, main = "Curva Precision-Recall - GBM", col = "red")
  legend("bottomright", legend = paste("AUC =", round(gbm_pr$auc.integral, 2)), col = "red", lty = 1, cex = 0.8)
} else {
  print("A classe escolhida não possui duas classes distintas em test_data$procedimento.")
}
