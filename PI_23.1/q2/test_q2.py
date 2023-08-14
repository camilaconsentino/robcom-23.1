import q2
import cv2

resposta_correta = [
    (4, 4, 3, 3),
    (3, 1, 2, 1),
    (3, 2, 2, 1),
    (3, 1, 3, 1)
]


def test_contagem_produtos():
    for i in range(1, 5):
        img = cv2.imread(f'img/teste{i}.png')
        n1, _, n2, _ = q2.prateleira_arrumada(img)
        assert n1 == resposta_correta[i -
                                      1][0], f'Número incorreto de produtos na prateleira de cima na imagem teste{i}'
        assert n2 == resposta_correta[i -
                                      1][2], f'Número incorreto de produtos na prateleira de baixo na imagem teste{i}'


def test_prateleira_incorreta():
    img = cv2.imread(f'img/teste3.png')
    n1, a1, n2, a2 = q2.prateleira_arrumada(img)
    assert n1 == resposta_correta[2][0], f'Número incorreto de produtos na prateleira de cima na imagem teste3'
    assert a1 == resposta_correta[2][1], f'Número incorreto de produtos arrumados na prateleira de cima na imagem teste3'
    assert n2 == resposta_correta[2][2], f'Número incorreto de produtos na prateleira de baixo na imagem teste3'
    assert a2 == resposta_correta[2][3], f'Número incorreto de produtos arrumados na prateleira de baixo na imagem teste3'


def test_orientacao_incorreta():
    img = cv2.imread(f'img/teste4.png')
    n1, a1, n2, a2 = q2.prateleira_arrumada(img)
    assert n1 == resposta_correta[3][0], f'Número incorreto de produtos na prateleira de cima na imagem teste4'
    assert a1 == resposta_correta[3][1], f'Número incorreto de produtos arrumados na prateleira de cima na imagem teste4'
    assert n2 == resposta_correta[3][2], f'Número incorreto de produtos na prateleira de baixo na imagem teste4'
    assert a2 == resposta_correta[3][3], f'Número incorreto de produtos arrumados na prateleira de baixo na imagem teste4'
