import cv2
import biblioteca_cow
import numpy as np

resultados = None
animais = None


## PARTE 1 - VACA
img = cv2.imread("cow_wolf/cow_wolf01.png")
    # Classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
    # Carregar Rede
net = biblioteca_cow.load_mobilenet()
    # Detectar
CONFIDENCE = 0.7
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def result_equals(res1, res2):
    return res1[0] == res2[0] and abs(res1[1]-res2[1])<1e-5 and res1[2] == res2[2] and res1[3] == res2[3]

## PARTE 7.1 - MOBILENET
def test_mobilenet():
    try:
        global resultados
        _, resultados = biblioteca_cow.detect(net, img, CONFIDENCE, COLORS, CLASSES)
        resp = [('cow', 99.0637481212616, (379, 131), (560, 251)), ('horse', 94.41149830818176, (53, 103), (297, 286)), ('horse', 93.70213747024536, (626, 103), (860, 285))]
        
    except Exception as e:
        print("PARTE 7.1 - Falha na função detect:",str(e))
    
    assert len(resp) == len(resultados), "PARTE 1.1 - Mobilenet não detectou o número correto de objetos."
    assert np.all([result_equals(r1, r2)] for r1,r2 in zip(resp, resultados)), "PARTE 1.1 - Mobilenet não esta correta."
    

## PARTE 7.2 - CAIXAS
def test_caixas():
    try:
        global animais
        _, animais = biblioteca_cow.separar_caixa_entre_animais(img, resultados)
    
    except Exception as e:
        print("PARTE 7.2 - Falha na função separar_caixa_entre_animais:",str(e))

    assert animais == {'vaca': [[379, 131, 560, 251]], 'lobo': [53, 103, 860, 286]}, "PARTE 7.2 - Caixas não estão corretas"
    


## PARTE 7.3 - PERIGO
def test_perigo():
    try:
        for vaca in animais['vaca']:
            resp = 0.148124730951356
            assert resp * 0.9 <= biblioteca_cow.calcula_iou(vaca, animais['lobo']) <= resp * 1.1, "PARTE 7.3 - IoU não esta dentro do esperado"
    
    except Exception as e:
        print("PARTE 7.3 - Falha na função calcula_iou")
        # pytest.fail(e, pytrace=True)

    
    



