import sys
import argparse

import cv2
import asone
from asone import ASOne
import torch

from asone import utils

def point_side_of_line(line, point):
    """Determina de que lado da linha um ponto se encontra usando o produto vetorial.
    Retorna:
    > 0 se o ponto está à esquerda da linha
    = 0 se o ponto está na linha
    < 0 se o ponto está à direita da linha
    """
    # vetor da linha: (x1, y1) to (x2, y2)
    line_vector = (line[2] - line[0], line[3] - line[1])

    # vetor do ponto inicial da linha para o ponto
    point_vector = (point[0] - line[0], point[1] - line[1])

    # produto vetorial
    return line_vector[0] * point_vector[1] - line_vector[1] * point_vector[0]


def main(args):
    filter_classes = args.filter_classes

    if filter_classes:
        filter_classes = ['person']

    # Verifica se o cuda está disponível
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False
     
    if sys.platform.startswith('darwin'):
        #Para rodar no Mac
        detector = asone.YOLOV7_MLMODEL 
    else:
        detector = asone.YOLOX_DARKNET_PYTORCH
    
    detect = ASOne(
        tracker=asone.BYTETRACK,
        detector=detector,
        weights=args.weights,
        use_cuda=args.use_cuda
        )
    
    # Obtenha a função de rastreamento
    track = detect.track_video(args.video_path,
                                output_dir=args.output_dir,
                                conf_thres=args.conf_thres,
                                iou_thres=args.iou_thres,
                                display=args.display,                                
                                filter_classes=filter_classes,
                                class_names=None) # class_names=['License Plate'] para pesos personalizados
       


    # linha horizontal no meio da imagem
    divisoria = [0, 360, 1280, 360]  

    count_direction = {
        "norte": 0,
        "sul": 0
    }

    #mapa que irá armazenar o sentido de cada veiculo detectado
    id_to_last_side = {}

    video_writer = cv2.VideoWriter(
                "./results/test2.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                29.97,
                (int(1280), int(720)),
    )                
    for bbox_details, frame_details in track:
        bbox_xyxy, ids, scores, class_ids = bbox_details   

        id_to_last_side_old = id_to_last_side.copy()
        
         # Código de processamento das frames de detecção ...
        for bbox, vehicle_id in zip(bbox_xyxy, ids):
            
            # Calculando o centro do bbox
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2

            # Determinar de que lado da divisória o veículo está
            side = point_side_of_line(divisoria, (x_center, y_center))

            # Verificar se o veículo cruzou a divisória
            if vehicle_id in id_to_last_side:
                #print(id_to_last_side[vehicle_id])

                if id_to_last_side[vehicle_id] * side <= 0:  # Mudança de sinal indica cruzamento
                    # O veículo cruzou a divisória                
                    direction = "sul" if side < 0 else "norte"
                    count_direction[direction] = count_direction.get(direction, 0) + 1
                    print(f"Veículo {vehicle_id} cruzou a divisória sentido {direction}!")

            # Atualizar a última posição conhecida            
            id_to_last_side[vehicle_id] = side

        frame, frame_num, fps = frame_details        

        frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids, identities=ids, class_names=None, draw_trails=True)

        # Desenhar a divisória
        cv2.line(frame, (divisoria[0], divisoria[1]), (divisoria[2], divisoria[3]), (0, 0, 255), 2)

        # Desenhar o número de veículos que cruzaram a divisória no canto superior direito
        directions = list(count_direction.keys())
        start_y = 50
        y_offset = 50

        for index, direction in enumerate(directions):
            text = f"{direction}: {count_direction.get(direction, 0)}"
            y_position = start_y + index * y_offset
            cv2.putText(frame, text, (1000, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow('Window', frame)
        video_writer.write(frame)
    
        print(frame_num)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Caminho para o vídeo de entrada')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda',
                        help='executar no cpu se não for fornecido o programa será executado no gpu.')
    parser.add_argument('--no_save', default=True, action='store_false',
                        dest='save_result', help='se salvar ou não os resultados')
    parser.add_argument('--no_display', default=True, action='store_false',
                        dest='display', help='se exibir ou não os resultados na tela')
    parser.add_argument('--output_dir', default='data/results',  help='Caminho para o diretório de saída')
    parser.add_argument('--filter_classes', default=None, help='Nome da classe de filtro')
    parser.add_argument('-w', '--weights', default=None, help='Caminho dos pesos treinados')
    parser.add_argument('-ct', '--conf_thres', default=0.25, type=float, help='limiar de pontuação de confiança')
    parser.add_argument('-it', '--iou_thres', default=0.45, type=float, help='limiar de pontuação iou')

    args = parser.parse_args()

    main(args)