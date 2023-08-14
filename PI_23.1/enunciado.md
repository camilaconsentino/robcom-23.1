# Robótica Computacional 2023.1

EMAIL: ______________

Observações de avaliações nesta disciplina:

* Inicie a prova no Blackboard para a ferramenta de Proctoring iniciar. Só finalize o Blackboard quando enviar a prova via Github classroom
* Durante esta prova vamos registrar somente a tela, não a câmera nem microfone
* Ponha o nome no enunciado da prova no Github
* Você pode consultar a internet ou qualquer material que usamos no curso, mas não pode se comunicar com pessoas ou colegas a respeito da prova. Também não pode usar ferramentas de **IA** como chatGPT ou Github Copilot durante a prova.
* Faça commits e pushes frequentes no seu repositório
* Avisos importantes serão dados na sala da prova
* Permite-se consultar qualquer material online ou próprio. Não se pode compartilhar informações com colegas durante a prova.
* Faça commits frequentes. O primeiro a enviar alguma ideia será considerado autor original
* A responsabilidade por ter o *setup* funcionando é de cada estudante
* Questões de esclarecimento geral podem ser perguntadas
* É vedado colaborar ou pedir ajuda a colegas ou qualquer pessoa que conheça os assuntos avaliados nesta prova.

## Questão 1 (3,5)

Essa questão consiste em identificar bandeiras de vários países em imagens. Um processo anterior já removeu o fundo, então só nos resta dizer qual é qual. Iremos analisar os seguintes países. Veja na pasta `q1/img` exemplos de todas essas bandeiras nas imagens de teste.

1. Mônaco
2. Peru
3. Singapura
4. Irlanda
5. Itália

Você deverá editar o arquivo `q1.py` para realizar essa questão. Neste arquivo, você deve modificar a função `identifica_bandeiras(imagem_bgr)`. Essa função deverá devolver uma lista com tuplas no seguinte formato:

`('pais', (x1, y2), (x2, y2)`)

onde

- `'pais'` é uma string com o nome do pais tratado (em minúsculas e sem espaços). Se você não conseguiu identificar a bandeira, pode enviar string vazia neste item.
- `(x1, y1)` é o ponto do topo esquerdo do retângulo em que a bandeira está inserida
- `(x2, y2)` é o ponto baixo direito do retângulo em que a bandeira está inserida

**A ordem dos elementos da lista não é importante, apenas seu conteúdo**

Os critérios de avaliação são os seguintes:

* **1,0** encontrou o canto de todas as bandeiras
* **1,9** identificou uma bandeira e acertou seus cantos em todas imagens de testes do arquivo `test_simples.py`
* **+0.4** para cada bandeira acertada corretamente em todas as imagems de testes do arquivo `test_simples.py`


## Questão 2 (3,5)

Estamos criando um programa de supermercado para verificar se os produtos estão posicionados corretamente. Cada produto é marcado por uma etiqueta retangular. Para garantir que os produtos estão posicionados corretamente precisamos checar se sua etiqueta está na orientação correta e se estão na prateleira correta.

- Etiqueta amarela: primeira prateleira, posição vertical
- Etiqueta magenta: segunda prateleira, posição horizontal

Nas imagens analisadas, a primeira prateleira está na metade de cima da imagem. Você deve preencher a função `prateleira_arrumada` para devolver quatro números

- número de produtos na prateleira de cima
- número de produtos que estão arrumados (prateleira correta e orientação correta) na prateleira de cima
- número de produtos na prateleira de baixo
- número de produtos que estão arrumados (prateleira correta e orientação correta) na prateleira de baixo

Os critérios de avaliação são os seguintes:

- **1,5** Conta quantos objetos de interesse em cada prateleira corretamente para todas imagens de teste
- **+1,0** Identifica corretamente produtos fora da prateleira correta em todas imagens de teste
- **+1,0** Identifica corretamente produtos fora da posição correta em todas imagens de teste

## Questão 3 (3,0)

Iremos desenvolver um sistema de monitoramento para um hotel de pets e precisamos garantir que animais não compatíveis fiquem perto demais uns dos outros. 

- gatos estão em perigo se ficarem perto de outros gatos e de cachorros
- pássaros podem ficar juntos ou perto de cachorros, mas não de gatos
- cachorros estão em perigo se estiverem próximos de outros dois cachorros

**Um animal está próximo de outro se seus centros estão a menos de 300 pixels de distância**

Essa questão está dividida em duas partes:

- **1,5** - preencha as funções `carregar` e `identificar_animais`. A primeira carrega a Mobile net. A segunda recebe uma imagem e devolve um dicionário na forma abaixo.

```python
{
    'passaros': [(x1, y1, x2, y2),  ....] ,# lista de posições em que foram encontrados pássaros
    'cachorros': [(x1, y1, x2, y2),  ....] ,# lista de posições em que foram encontrados cachorros
    'gatos': [(x1, y1, x2, y2),  ....]# lista de posições em que foram encontrados gatos
}
```

- **1,5** - crie uma função `lista_perigos` que, dado um dicionário de animais como mostrado acima, retorna um dicionário somente com os animais em perigo. Ele deve estar no mesmo formato acima, mas conter somente os animais que satisfaçam o critério do início da função.