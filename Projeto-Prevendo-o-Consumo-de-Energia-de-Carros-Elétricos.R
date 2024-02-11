          ## MACHINE LEARNING EM LOGÍSTICA ##
## PREVENDO O CONSUMO DE ENERGIA DE CARROS ELÉTRICOS ##

# Carregando os pacotes necessários
library(dplyr)
library(readxl)
library(ggplot2)
library(corrplot)
library(randomForest)
library(caret)
library(openxlsx)

# Carregando o dataset
dados <- read_excel('FEV-data-Excel.xlsx')

# Conhecendo os dados
dim(dados)
View(dados)
str(dados)

# Sumário das variáveis numéricas
summary(dados)

# Análise Exploratória dos dados - Limpeza

# Observações completas
completos <- sum(complete.cases(dados))

# Observações incompletas
incompletos <- sum(!complete.cases(dados))

# Representatividade dos casos incompletos
percentual <- (incompletos / completos) * 100
percentual

# Visto que o percentual de casos incompletos é alto em relação ao tamanho da
# amostra, vou preencher as linhas faltantes com a mediana. Não acredito que
# a média seria adequada nesse caso, pois a maioria das variáveis tem um desvio
# padrão relativamente alto.
dados <- dados %>%
  mutate_all(funs(ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Ainda resta 1 caso com valor NA. Nesse caso, vou removê-lo
dados <- na.omit(dados)

# Aplicando One-Hot-Encoding nas variáveis categóricas
categoricas <- sapply(dados, is.character)
dados <- dados %>%
  mutate_if(categoricas, as.factor)

str(dados)

# Alterando os nomes das colunas
colunas_ingles <- colnames(dados)

colunas_ingles[1] <- 'Nome_carro'
colunas_ingles[2] <- 'Marca'
colunas_ingles[3] <- 'Modelo'
colunas_ingles[4] <- 'Preço_minimo_bruto'
colunas_ingles[5] <- 'Potencia_motor'
colunas_ingles[6] <- 'Torque_maximo'
colunas_ingles[7] <- 'Freios'
colunas_ingles[8] <- 'Tipo_direcao'
colunas_ingles[9] <- 'Capacidade_bateria'
colunas_ingles[10] <- 'Autonomia'
colunas_ingles[11] <- 'Dist_entre_eixos'
colunas_ingles[12] <- 'Comprimento'
colunas_ingles[13] <- 'Largura'
colunas_ingles[14] <- 'Altura'
colunas_ingles[15] <- 'Peso_minimo_vazio'
colunas_ingles[16] <- 'Peso_bruto_permitido'
colunas_ingles[17] <- 'Capacidade_maxima_carga'
colunas_ingles[18] <- 'Quantidade_assentos'
colunas_ingles[19] <- 'Num_portas'
colunas_ingles[20] <- 'Aro_pneu'
colunas_ingles[21] <- 'Velocidade_maxima'
colunas_ingles[22] <- 'Capacidade_inicializacao'
colunas_ingles[23] <- 'Aceleracao_0_100'
colunas_ingles[24] <- 'Potencia_max_carregamento'
colunas_ingles[25] <- 'Consumo_medio'

colnames(dados) <- colunas_ingles
rm(colunas_ingles)
View(dados)

# Análise Exploratória - Conhecendo os dados

# Extraindo as variáveis numéricas
numericas <- sapply(dados, is.numeric)
dados_numericos <- dados[numericas]

# Matriz de correlação
matriz <- cor(dados_numericos)

# Plot de correlação
corrplot(matriz, method = 'square', type = 'upper', tl.cex = 0.6)

# Aparentemente, o consumo de médio está ligado positivamente à alguns fatores,
# como: potência, torque, dimensões e pesos (tanto do veículo quanto sua capacidade de carga).
names(dados)
ggplot(dados, aes(x = Consumo_medio, y = Potencia_motor)) +
  geom_point() +
  ggtitle("Relação entre Consumo e Potência do motor") +
  xlab('Consumo médio') +
  ylab('Potência')

ggplot(dados, aes(x = Consumo_medio, y = Torque_maximo)) +
  geom_point() +
  ggtitle('Relação entre Consumo e Torque') +
  xlab('Consumo médio') +
  ylab('Torque')

ggplot(dados, aes(x = Consumo_medio, y = Peso_minimo_vazio)) +
  geom_point() +
  ggtitle('Relação entre Consumo e Peso Mínimo do Veículo') +
  xlab('Consumo médio') +
  ylab('Peso Mínimo do veículo')

ggplot(dados, aes(x = Consumo_medio, y = Capacidade_maxima_carga)) +
  geom_point() +
  ggtitle('Relação entre Consumo e Capacidade de carga') +
  xlab('Consumo médio') +
  ylab('Capacidade de carga')

ggplot(dados, aes(x = Consumo_medio, y = Aceleracao_0_100)) +
  geom_point() +
  ggtitle('Relação entre Consumo e Aceleração') +
  xlab('Consumo') +
  ylab('Aceleração')

# Mas, vou aplicar Feature Selection (com Random Forest nos dados para
# me ajudar na seleção das variáveis com mais relevância para o modelo
# de regressão linear.
modelo_rf1 <- randomForest(Consumo_medio ~ .,
                        data = dados,
                        ntree = 100,
                        nodesize = 10,
                        importance = TRUE)

# Plotando as variáveis pelo grau de importância
varImpPlot(modelo_rf1)

# Machine Learning

# Dividindo os dados em treino e teste
split <- createDataPartition(dados$Consumo_medio, p = 0.65, list = FALSE)
treino <- dados[split,]
teste <- dados[-split,]

# Primeira versão do modelo de Regressão Linear
modelo1 <- lm(Consumo_medio ~ ., data = treino)
summary(modelo1)

# Segunda versão do modelo com as variáveis significativas
modelo2 <- lm(Consumo_medio ~ Peso_bruto_permitido +
                Capacidade_maxima_carga +
                Peso_minimo_vazio +
                Autonomia +
                Capacidade_bateria +
                Aceleracao_0_100 +
                Velocidade_maxima,
              data = treino)
summary(modelo2)

# Terceira versão do modelo com mais redução de variáveis
modelo3 <- lm(Consumo_medio ~ Peso_bruto_permitido +
                Autonomia +
                Capacidade_bateria,
              data = treino)
summary(modelo3)

# Fazendo as previsões
# Verifiquei que o modelo2 apresentou uma diferença menor no teste
teste$Consumo_medio_previsto <- predict(modelo2, newdata = teste)
View(teste)
teste$Diferenca <- teste$Consumo_medio - teste$Consumo_medio_previsto
sum(teste$Diferenca)

# Exportando o resultado
write.xlsx(teste, file = "Previsoes.xlsx")

# Fim
