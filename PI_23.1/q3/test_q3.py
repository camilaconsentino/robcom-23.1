import cv2
import q3

def retangulo_igual(p1r1, p2r1, p1r2, p2r2):
    diff = 0
    for p in [(p1r1[0], p1r2[0]), (p1r1[1], p1r2[1]), (p2r1[0], p2r2[0]), (p2r1[1], p2r2[1])]:
        diff += abs(p[0] - p[1])

    return diff < 20


resultados_corretos_parte1 = [
    {'gatos': [(899, 478, 1132, 824)], 'cachorros': [(209, 449, 558, 811)], 'passaros': [(715, 149, 909, 402)]},
    {'gatos': [(700, 114, 917, 447), (983, 111, 1207, 463), (373, 397, 605, 734)], 'cachorros': [(132, 454, 333, 813)], 'passaros': []},
    {'gatos': [(473, 191, 685, 536)], 'cachorros': [], 'passaros': [(748, 238, 965, 519)]},
    {'gatos': [], 'cachorros': [(865, 510, 1224, 881), (170, 127, 517, 492)], 'passaros': []},
    {'gatos': [], 'cachorros': [(722, 241, 1080, 607), (471, 349, 790, 714), (202, 475, 527, 851)], 'passaros': []},
    {'gatos': [], 'cachorros': [], 'passaros': [(820, 576, 1029, 796), (905, 185, 1123, 407), (302, 174, 519, 394)]}
]

def test_parte1():
    res = q3.carrega()
    for i in range(1, 7):
        img = cv2.imread(f'img/teste{i}.png')
        d = q3.identificar_animais(res, img)

        for k, v in resultados_corretos_parte1[i-1].items():
            for r in v:
                tem_ret = False
                for r2 in d[k]:
                    if retangulo_igual((r[0], r[1]), (r[2], r[3]), (r2[0], r2[1]), (r2[2], r2[3])):
                        tem_ret = True
                assert tem_ret, f'Problema na imagem teste{i}'
    
resultados_corretos_parte2 = [
    {'gatos': [], 'cachorros': [], 'passaros': []},
    {'gatos': [(700, 114, 917, 447), (983, 111, 1207, 463), (373, 397, 605, 734)], 'cachorros': [], 'passaros': []},
    {'gatos': [], 'cachorros': [], 'passaros': [(748, 238, 965, 519)]},
    {'gatos': [], 'cachorros': [], 'passaros': []},
    {'gatos': [], 'cachorros': [(471, 349, 790, 714)], 'passaros': []},
    {'gatos': [], 'cachorros': [], 'passaros': []}
]

def test_parte2():
    for i in range(6):
        d_update = q3.lista_perigos(resultados_corretos_parte1[i])

        for k, v in resultados_corretos_parte2[i].items():
            assert len(d_update[k]) == len(v), f'Número de {k} diferentes entre resultado e saída esperada'
            for r in v:
                tem_ret = False
                for r2 in d_update[k]:
                    if retangulo_igual((r[0], r[1]), (r[2], r[3]), (r2[0], r2[1]), (r2[2], r2[3])):
                        tem_ret = True
                assert tem_ret, f'Problema na imagem teste{i}'

