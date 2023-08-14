import cv2
import q1


resultados_corretos = [
    [
        ('singapura', (192, 496), (456, 673)),
        ('monaco', (726, 163), (983, 369)),
        ('peru', (119, 121), (380, 295)),
    ],
    [
        ('irlanda', (705, 589), (970, 722)),
        ('italia', (343, 298), (607, 474)),
    ],
    [
        ('peru', (751, 445), (1012, 619)),
        ('singapura', (125, 261), (390, 437)),
    ],
    [
        ('peru', (767, 496), (1028, 671)),
        ('italia', (84, 477), (348, 653)),
        ('irlanda', (752, 114), (1017, 246)),
    ]
]


def retangulo_igual(p1r1, p2r1, p1r2, p2r2):
    diff = 0
    for p in [(p1r1[0], p1r2[0]), (p1r1[1], p1r2[1]), (p2r1[0], p2r2[0]), (p2r1[1], p2r2[1])]:
        diff += abs(p[0] - p[1])

    return diff < 20


def test_todas_posicoes_corretas():
    for i in range(1, 5):
        img = cv2.imread(f'img/teste{i}.png')
        bandeiras = set(q1.identifica_bandeiras(img.copy()))
        correto = set(resultados_corretos[i-1])

        assert len(correto) == len(
            bandeiras), f'Número incorreto de bandeiras identificado em teste{i}.png'

        for v in correto:
            p1, p2 = v[1], v[2]
            encontrou = False
            for b in bandeiras:
                encontrou = encontrou or retangulo_igual(p1, p2, b[1], b[2])

            assert encontrou, f'Falhou em encontrar bandeira na imagem teste{i}.png'


def encontrou_bandeira(correto, bandeiras, i, bandeira_atual):
    for v in correto:
        if v[0] != bandeira_atual:
            continue

        p1, p2 = v[1], v[2]
        encontrou = False
        for b in bandeiras:
            encontrou = encontrou or (retangulo_igual(
                p1, p2, b[1], b[2]) and b[0] == bandeira_atual)

        assert encontrou, f'Falhou em encontrar {bandeira_atual} na imagem teste{i}.png'


def test_encontra_singapura():
    for i in range(1, 5):
        img = cv2.imread(f'img/teste{i}.png')
        bandeiras = set(q1.identifica_bandeiras(img.copy()))
        correto = set(resultados_corretos[i-1])

        encontrou_bandeira(correto, bandeiras, i, 'singapura')


def test_encontra_monaco():
    for i in range(1, 5):
        img = cv2.imread(f'img/teste{i}.png')
        bandeiras = set(q1.identifica_bandeiras(img.copy()))
        correto = set(resultados_corretos[i-1])

        encontrou_bandeira(correto, bandeiras, i, 'monaco')


def test_encontra_peru():
    for i in range(1, 5):
        img = cv2.imread(f'img/teste{i}.png')
        bandeiras = set(q1.identifica_bandeiras(img.copy()))
        correto = set(resultados_corretos[i-1])

        encontrou_bandeira(correto, bandeiras, i, 'peru')


def test_encontra_italia():
    for i in range(1, 5):
        img = cv2.imread(f'img/teste{i}.png')
        bandeiras = set(q1.identifica_bandeiras(img.copy()))
        correto = set(resultados_corretos[i-1])

        encontrou_bandeira(correto, bandeiras, i, 'italia')


def test_encontra_irlanda():
    for i in range(1, 5):
        img = cv2.imread(f'img/teste{i}.png')
        bandeiras = set(q1.identifica_bandeiras(img.copy()))
        correto = set(resultados_corretos[i-1])

        encontrou_bandeira(correto, bandeiras, i, 'irlanda')

