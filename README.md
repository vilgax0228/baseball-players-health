## Introdução: Do Baseball Players Gain Weight as they Age?
Atletas se esforçam para manter a forma física. No entanto, mesmo eles podem ganhar peso ao longo do tempo, assim como acontece com a população em geral. Até que ponto isso ocorre com os jogadores de beisebol?

Com o dataset disponibilizado pelo *UCLA Statistics Department*, nosso objetivo aqui é investigar através da análise exploratória dos dados, as seguintes perguntas:

1. **Construir um modelo de previsão.** Ao trabalhar com grandes bases de dados é natural que hajam valores faltantes, para isso podemos usar de um modelo de previsão para, por exemplo, prever o peso ou altura de um determinado jogador que não esteja disponível.

2. **Jogadores profissionais de beisebol ganham peso ao longo dos anos?** it will turn out that by trying to predict weight, we can deduce effects of height and age. In particular, we can answer the question posed above concerning weight gain over time.

## Bibliotecas e Módulos Importados
Para utilizar o dataset analisado é necessário instalar e carregar o pacote **freqparcoord**, após isso carregue o dataset **mlb**.

```R
install.packages("freqparcoord")
library(freqparcoord)
library(dplyr)
data(mlb)
```

## Processamento
As variáveis de interesse são peso, altura e idade, especialemnte as duas primeiras. Portanto, vamos renomear tais colunas.

```R
mlb = mlb %>%
  rename(Altura = Height, Peso = Weight, Idade = Age)
head(mlb)
             Name Team       Position Altura Peso Idade PosCategory
1   Adam_Donachie  BAL        Catcher     74  180 22.99     Catcher
2       Paul_Bako  BAL        Catcher     74  215 34.69     Catcher
3 Ramon_Hernandez  BAL        Catcher     72  210 30.78     Catcher
4    Kevin_Millar  BAL  First_Baseman     72  210 35.43   Infielder
5     Chris_Gomez  BAL  First_Baseman     73  188 35.71   Infielder
6   Brian_Roberts  BAL Second_Baseman     69  176 29.39   Infielder
```

Verificando se há qualquer valor faltante ou duplicado
```R
anyNA(mlb)
[1] FALSE
anyDuplicated(mlb)
[1] 0
```

> [!TIP]
> 72 polegadas de altura correspondem a 6 pés ou 183 cm.


## Conceitos e Etimologias
A diferença entre amostra e população está, é claro, no cerne da estatística.

Desejamos calcular o peso para toda população (nesse caso, jogadores de beisebol), porém o *dataset é uma amostra* de apenas 1015 observações dessa mesma população. Portanto, o que devemos fazer aqui é **estimar** os valores com o qual queremos trabalhar.

E() refere-se à *média populacional*, então se P é para peso, E(P) não é apenas a média dessa variável P, mas, mais importante, é o *peso médio de todas as pessoas nessa população*.

Da mesma forma, podemos definir *médias condicionais*, ou seja, médias de subpopulações.

Então, suponha que teremos um fluxo contínuo de jogadores para os quais apenas conhecemos a altura (A) e precisamos prever seus pesos. Usaremos a média condicional para fazer isso. Para um jogador com altura de 72 polegadas, por exemplo, nossa previsão pode ser: ^P = E(P | A = 72).

Novamente, no entanto, este é um valor populacional, e tudo o que temos são dados de amostra. Como estimaremos E(P | A = 72) a partir desses dados?

µ (mu) é a letra grega para denotar a média populacional, vamos agora usá-la para representar uma função que nos fornece as médias de subpopulações:

Para qualque altura t, define: µ(t) = E(P | A = t). Que é o peso médio de todas as pessoas da população que têm altura t.

Já que podemos variar t, temos uma função, e é conhecida como a função de regressão de P em A.

Então, µ(72.12) é a peso médio populacional de todos os jogadores de 72.12 de altura. Essas médias são valores populacionais e, portanto, desconhecidos, mas elas existem.

Portanto, para prever o peso de um jogador de 71.6 polegadas de altura, usaríamos µ(71.6) se soubéssemos esse valor, o que não sabemos, já que, mais uma vez, este é um valor populacional, enquanto temos apenas dados de amostra.

Precisamos estimar esse valor a partir dos pares de dados (altura, peso) em nossa amostra, que denotaremos como (A1, A1), ...(A1015, A1015).

Os dados de altura são medidos apenas para a polegada mais próxima, então, em vez de estimar valores como 𝜇(71.6), vamos optar por 𝜇(72) e assim por diante.

Uma estimativa natural para 𝜇(72), novamente usando o símbolo "chapeuzinho" para indicar "estimativa de", é a média de peso entre todos os jogadores em nossa amostra para os quais a altura é 72.

^µ(72) = média de todos os Pi tal que Ai = 72.

**Observações importantes:**

1. A estratégia que consiste em "tirar" a média da subpopulação é otimizada já que minimiza nosso erro quadrático médio esperado de previsão;

2. Não foi feito nenhuma forma de feature selection/importance por se tratar de um dataset simples, porém é bom lembrar que apenas isso não indica que há causalidade ou qualquer tipo de relação entre a variável resposta e as variáveis selecionadas.

## Análise
O dataset têm 1015 linhas e 7 colunas.
```R
str(mlb)
'data.frame':	1015 obs. of  7 variables:
```

*tapply()* pode nos dar todos os ^µ(t) de uma vez:
```R
muhats = tapply(mlb$Peso, mlb$Altura, mean)
muhats
      67       68       69       70       71 
172.5000 173.8571 179.9474 183.0980 190.3596 
      72       73       74       75       76 
192.5600 196.7716 202.4566 208.7161 214.1386 
      77       78       79       80       81 
216.7273 220.4444 218.0714 237.4000 245.0000 
      82       83 
240.5000 260.0000
```

Pedimos ao R para dividir a variável Peso em grupos de acordo com os valores da variável Altura e, em seguida, calcular a média de peso em cada grupo.

Logo, a média de peso das pessoas com altura de 72 na nossa amostra foi 192,5600. Em outras palavras, ^𝜇(72) = 192.5600, ^𝜇(74) = 202.4566, e assim por diante.

Como estamos simplesmente realizando a operação estatística elementar de estimar médias populacionais a partir de amostras, podemos formar intervalos de confiança (ICs).

Para isso, precisaremos do "n" (tamanho da amostra) e do desvio padrão amostral para cada grupo de altura.

```R
n = tapply(mlb$Peso, mlb$Altura, length)
 67  68  69  70  71  72  73  74  75  76  77  78  79 
  2   7  19  51  89 150 162 173 155 101  55  27  14 
 80  81  82  83 
  5   2   2   1

sd = tapply(mlb$Peso, mlb$Altura, sd)
      67       68       69       70       71 
10.60660 22.08641 15.32055 13.54143 16.43461 
      72       73       74       75       76 
17.56349 16.41249 18.10418 18.27451 19.98151 
      77       78       79       80       81 
18.48669 14.44974 28.17108 10.89954 21.21320 
      82       83 
13.43503       NA
```

Lembre-se de que essa função divide os dados pela variável Altura, resultando em um vetor de pesos para cada valor de altura.

Já o desvio padrão (sd), então, nos fornece a contagem de pesos para cada valor de altura, o "n" que precisamos para formar um intervalo de confiança.

A propósito, o valor NA ocorre porque há apenas um jogador com altura 83, o que torna impossível o cálculo do desvio padrão (sd()), já que ele divide por "n-1".

Um intervalo de confiança (IC) aproximado de 95% para 𝜇(72), por exemplo, é então: ^𝜇 +- 1.96*sd/sqrt(n)

```R
ic_neg = 192.5600 - 1.96* 17.56349/sqrt(150)
ic_pos = 192.5600 + 1.96* 17.56349/sqrt(150)
ic_neg
[1] 189.7493
ic_pos
[1] 195.3707
```

within the data mining process it is customary to run several alternative learning algorithms in order to identify the most accurate model.

A análise acima adota o que é chamado de abordagem *não paramétrica*.

Até agora, não assumimos nada sobre a forma que 𝜇(𝑡) teria, caso fosse plotada em um gráfico. Novamente, ela é desconhecida, mas a função existe e, portanto, corresponde a alguma curva.

Mas podemos considerar fazer uma suposição sobre a forma dessa curva desconhecida. Isso pode parecer estranho, mas você verá abaixo que essa é uma ideia muito poderosa e intuitivamente razoável.

Com esse objetivo, vamos plotar os valores de 𝜇(𝑡) que encontramos acima.

```R
plot(67:83, muhats)
```

![Rplot05](https://github.com/user-attachments/assets/607907ca-b5d7-4207-ae3b-64d98860fdf8)

Os pontos neste gráfico parecem estar próximos de uma linha reta.

Isso sugere que nossa função desconhecida 𝜇(𝑡) tem uma forma linear, ou seja, que 𝜇(𝑡) = c + dt.

peso médio = c + d * altura.

Não se esqueça da palavra "média" aqui. Estamos assumindo que os pesos médios nas diversas subpopulações de altura têm a forma 𝜇(𝑡) = c + dt.

Isso é chamado de modelo paramétrico para 𝜇(𝑡) com os parâmetros 𝑐 e 𝑑. Vamos usar isso abaixo para estimar 𝜇(𝑡).

Como é 𝜇(𝑡) uma função populacional, as constantes 𝑐 e 𝑑 são valores populacionais, portanto desconhecidos. No entanto, podemos estimá-los a partir dos nossos dados amostrais. Fazemos isso usando a função *lm()*.

```R
modelo_linear = lm(formula = Peso ~ Altura, data = mlb)
coef(modelo_linear)
(Intercept)      Altura 
-151.133291    4.783332
```

Isso nos dá ^c = -151.133291 e ^d = 4.783332. Podemos sobrepor a linha ajustada para o gráfico plotado usando a função *abline()* que adiciona uma linha com a inclinação e o intercepto especificados ao gráfico atualmente exibido.

```R
abline(coef = coef(modelo_linear), col = "blue")
```

![Rplot06](https://github.com/user-attachments/assets/70ab5de6-3072-41cc-b522-f90e968de342)

^𝜇(72) = -151.133291 + 4.783332 * 72 = 193.2666.

Então, usando este modelo, nós preveríamos um peso um pouco mais alto do que nossa previsão anterior.

Calculando ^𝜇(72) por multiplicação de matrizes com o operador %*%.
```R
coef(modelo_linear) %*% c(1,72)
         [,1]
[1,] 193.2666
```

Podemos formar um intervalo de confiança a partir disso também, que para o nível de 95% será:

^𝜇(72) +- 1.96*se[(^𝜇(72)].

Onde se[] significa erro padrão, a estimativa do desvio padrão de um estimador.

Aqui ^𝜇 por ser baseado em nossos dados de amostra aleatória, é em si aleatório, ou seja, ele variará de uma amostra para outra. Assim, ele possui um desvio padrão, que chamamos de erro padrão.

```R
tmp = c(1,72)
sqrt(tmp %*% vcov(modelo_linear) %*% tmp)
          [,1]
[1,] 0.6859655
193.2666 - 1.96*0.6859655
[1] 191.9221
193.2666 + 1.96*0.6859655
[1] 194.6111
```

Assim, um IC aproximado de 95% para ^𝜇(72) nesse modelo seria em torno de (191,9, 194,6).

Agora, aqui está um ponto importante: O intervalo de confiança (IC) que obtivemos a partir do nosso modelo linear, (191,9, 194,6), foi mais estreito do que o fornecido pela abordagem não paramétrica, (187,6, 193,2); o primeiro tem uma largura de aproximadamente 2,7, enquanto o último é de 5,6. Em outras palavras:

Um modelo paramétrico é — se for (aproximadamente) válido — mais poderoso do que um modelo não paramétrico, fornecendo estimativas de uma função de regressão que tendem a ser mais precisas do que as obtidas pela abordagem não paramétrica. Isso deve se traduzir também em previsões mais precisas.

Por que o modelo linear deveria ser mais eficaz? Aqui está uma intuição: digamos que para estimar 𝜇(72): A função lm() usa todos os dados para estimar os coeficientes de regressão.

No nosso caso, todos os 1015 pontos de dados desempenharam um papel no cálculo de ^𝜇(72), enquanto apenas 150 de nossas observações foram usadas no cálculo da nossa estimativa não paramétrica ^𝜇(72). A primeira, por ser baseada em muito mais dados, *tende a ser* mais precisa.

---

Agora vamos prever o peso a partir da altura e da idade.


