import numpy as np
import pyautogui
import imutils
import cv2
import math
import keyboard
import time
from random import sample, shuffle


def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def move(Side, OldX, OldY, NewX, NewY, Boardin):
    Board = np.copy(Boardin)
    score = 0
    #pyautogui.click(x1+remap(OldX,0,8,0,x2-x1)+RektSize/2,y1+remap(OldY,0,8,0,y2-y1)+RektSize/2)
    #if Board[NewY,NewX]!="":
    #    score = Pieces[Board[NewY,NewX].split()[1]]
    if(len(Board[NewY,NewX].split())>1):
        score = Pieces[Board[NewY,NewX].split()[1]]*20
    if "Runner" in Board[OldY,OldX]:
        score=score+20
    if "King"  in Board[NewY,NewX]:
        score=float("inf")
    if "King"  in Board[OldY,OldX]:
        score=float(40)
    Board[NewY, NewX] = Board[OldY, OldX]
    Board[OldY, OldX] = ""
    x, y = Board.shape[::1]
   # if Side == "Black":
        #print("Opposide")
    for x in range(x):
        for y in range(y):
            if len(Board[y, x].split()) > 1:
                s = Board[y, x].split()
                if s[0] == Side:
                    score = score+Pieces[s[1]]
                else:
                    score = score-2*Pieces[s[1]]
    #pyautogui.click(x1+remap(NewX,0,8,0,x2-x1)+RektSize/2,y1+remap(NewY,0,8,0,y2-y1)+RektSize/2)
    return np.flipud(Board), score


Template = cv2.imread("template.png", 0)
TW, TH = Template.shape[::-1]
Colors = ["White", "Black"]
Pieces = {
    "Farmer": 1,
    "Runner": 6,
    "Queen": 9,
    "Horse": 3,
    "Tower": 5,
    "King": 1000
}
PieceImg = {}
def makemove(side = "White",lastScore = float("-inf"),score = -4,TempBoard = np.array((8,8)),Player = ((2,2),(2,2)),px = 0,py= 0,i = 0,depth = 3):
                        lastScore = score
                        if side == "White":
                            return  (((Player[1][1], Player[1][0]), (px, py),
                                              Pieces["Farmer"]*(score-SimMove("Black", TempBoard, depth, i,lastScore,score))))
                        else:
                            return (((Player[1][1], Player[1][0]), (px, py),
                                              Pieces["Farmer"]*(score-SimMove("White", TempBoard, depth, i,lastScore,score))))

def LoadPieces():

    for color in Colors:
        for piece in list(Pieces.keys()):
            im = cv2.imread(
                "{}{}.png".format(piece, color), 0)
            fac = (int(im.shape[1]*(x2-x1)/TW), int(im.shape[0]*(y2-y1)/(TH)))
            PieceImg["{}{}".format(piece, color)] = cv2.resize(im, fac)


board = []


def ReadBoard(Side):
    board = []
    for i in range(8):
        board.append([])
        for j in range(8):
            board[i].append("")
    imageColor = pyautogui.screenshot()
    imageColor = cv2.cvtColor(np.array(imageColor), cv2.COLOR_RGB2BGR)
    imageColor = cv2.cvtColor(np.array(imageColor), cv2.COLOR_BGR2RGB)
    imageColor[np.all(imageColor == (118, 150, 86), axis=-1)] = (238, 238, 210)
    imageColor = imageColor[y1:y2, x1:x2]
    image = cv2.cvtColor(np.array(imageColor), cv2.COLOR_BGR2GRAY)
    for color in Colors:
        for piece in list(Pieces.keys()):
            s = piece+color
            w, h = PieceImg[s].shape[::-1]
            res = cv2.matchTemplate(image, PieceImg[s], cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                x = int(math.floor(remap(pt[0], 0, x2-x1, 0, 8)))
                y = int(math.floor(remap(pt[1]+10, 0, y2-y1, 0, 8)))
                board[y][x] = "{} {}".format(color, piece)
                if(color == "White"):
                    cv2.rectangle(
                        imageColor, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(
                        imageColor, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)
    cv2.imwrite("in_memory_to_disk.png", imageColor)
    #if(board!=OldBoard):
    board = np.array(board)
    best = SimMove(Side, board,3 , 0)
    board = []
    bestX = int(remap(best[0], 0, 8, 0, x2-x1))
    bestY = int(remap(best[1], 0, 8, 0, y2-y1))
    cv2.rectangle(
                        imageColor, pt, (bestX + 10, bestY + 10), (0, 0, 255), 2)
    cv2.imshow("Screenshot", imutils.resize(imageColor, width=600))
    cv2.waitKey(0)


Side = ""


def SimMove(side, Board, depth, i):
    if i >= depth:
        return 0
    else:
        Board = np.array(Board)
        BoardTemp = np.copy(Board)
        if(side == "White"):
            opposide = "Black"
            king = np.where(BoardTemp == "Black King")
        else:
            opposide = "White"
            king = np.where(BoardTemp == "White King")
        score = 0
        lastScore = 0
        i = i+1
        FriendlyPlayers = []
        EnemyPlayers = []
        X, Y = BoardTemp.shape[::1]
        for x in range(X):
            for y in range(Y):
                if len(BoardTemp[y][x].split()) > 1:
                    if BoardTemp[y][x].split()[0] == side:
                            FriendlyPlayers.append(
                                (BoardTemp[y][x].split()[1], (y, x)))
                    else:
                            EnemyPlayers.append(
                                (BoardTemp[y][x].split()[1], (y, x)))
        movescore = []
        moves = []
        shuffle(FriendlyPlayers)
        for Player in FriendlyPlayers:
            if(Player[0] == "Farmer"):
                
                px = Player[1][1]
                py = Player[1][0]
                if py == 0:
                    time.sleep(2)
                if px-1 >= 0 and  opposide in BoardTemp[py-1, px-1]:
                    moves.append(((px,px),(px-1,py-1),Player[0]))
                if px+1 <= 7 and  opposide in BoardTemp[py-1, px+1]:
                    moves.append(((px,px),(px+1,py-1),Player[0]))
                if py > 0 and BoardTemp[py-1, px] == "":
                    moves.append(((px,px),(px,py-1),Player[0]))
            elif(Player[0] == "Runner"):
                px = Player[1][1]
                py = Player[1][0]
                while px+1 <= 7 and py+1 <= 7 and not side in BoardTemp[py+1, px+1]  :
                    moves.append(((px,px),(px+1,py+1),Player[0]))
                    px = px+1
                    py = py+1
                    if opposide in BoardTemp[py,px]:
                        break
                px = Player[1][1]
                py = Player[1][0]
                while px-1 >= 0 and py+1 <= 7 and not side in BoardTemp[py+1, px-1] :
                    moves.append(((px,px),(px-1,py+1),Player[0]))
                    px = px-1
                    py = py+1
                    if opposide in BoardTemp[py,px]:
                        break
                px = Player[1][1]
                py = Player[1][0]
                while px+1 <= 7 and py-1 >= 0 and not side in BoardTemp[py-1, px+1]:
                    moves.append(((px,px),(px+1,py-1),Player[0]))
                    px = px+1
                    py = py-1
                    if opposide in BoardTemp[py,px]:
                        break
                px = Player[1][1]
                py = Player[1][0]
                while px-1 >= 0 and py-1 >= 0 and side not in BoardTemp[py-1, px-1] :
                    moves.append(((px,px),(px-1,py-1),Player[0]))
                    px = px-1
                    py = py-1
                    if opposide in BoardTemp[py,px]:
                        break
            elif(Player[0] == "Horse"):
                px = Player[1][1]
                py = Player[1][0]
                if px-2 >= 0 and py-1 >= 0 and side not in BoardTemp[py-1, px-2]:
                    moves.append(((px,px),(px-2,py-1),Player[0]))
                if px-2 >= 0 and py+1 <= 7 and side not in BoardTemp[py+1, px-2]:
                    moves.append(((px,px),(px-2,py+1),Player[0]))
                if px+2 <= 7 and py-1 >= 0 and side not in BoardTemp[py-1, px+2]:
                    moves.append(((px,px),(px+2,py-1),Player[0]))
                if px+2 <= 7 and py+1 <= 7 and side not in BoardTemp[py+1, px+2]:
                    moves.append(((px,px),(px+2,py+1),Player[0]))
                # Vertikala förflyttningar
                if px-1 >= 0 and py-2 >= 0 and side not in BoardTemp[py-2, px-1]:
                    moves.append(((px,px),(px-1,py-2),Player[0]))
                if px < 7 and py-2 >= 0 and side not in BoardTemp[py-2, px+1]:
                    moves.append(((px,px),(px+1,py-2),Player[0]))
                if px > 0 and py+2 <= 7 and not side in BoardTemp[py+2, px-1]:
                    moves.append(((px,px),(px-1,py+2),Player[0]))
                if px+1 <= 7 and py+2 <= 7 and side not in BoardTemp[py+2, px+1]:
                    moves.append(((px,px),(px+1,py+2),Player[0]))
            elif(Player[0] == "Tower"):
                px = Player[1][1]
                py = Player[1][0]
                while px < 7 and not side in BoardTemp[py, px+1] and not opposide in BoardTemp[py,px]:
                    moves.append(((px,px),(px+1,py),Player[0]))
                    px = px+1
                    if opposide in BoardTemp[py,px]:
                        break
                px = Player[1][1]
                py = Player[1][0]
                while px > 0 and not side in BoardTemp[py, px-1] and not opposide in BoardTemp[py,px]:
                    moves.append(((px,px),(px-1,py),Player[0]))
                    px = px-1
                    if opposide in BoardTemp[py,px]:
                        break
                px = Player[1][1]
                py = Player[1][0]
                while py > 0 and not side in BoardTemp[py-1, px] and not opposide in BoardTemp[py,px]:
                    moves.append(((px,px),(px,py-1),Player[0]))
                    py = py-1
                    if opposide in BoardTemp[py,px]:
                        break
                px = Player[1][1]
                py = Player[1][0]
                while py < 7 and not side in BoardTemp[py+1, px] and not opposide in BoardTemp[py,px]:
                    moves.append(((px,px),(px,py+1),Player[0]))
                    py = py+1
                    if opposide in BoardTemp[py,px]:
                        break
            elif(Player[0] == "King"):
                px = Player[1][1]
                py = Player[1][0]
                if py+1 < 8 and not side in BoardTemp[py+1, px]:
                    moves.append(((px,px),(px,py+1),Player[0]))
                if py > 0 and not side in BoardTemp[py-1, px]:
                    moves.append(((px,px),(px,py-1),Player[0]))
                if py < 7 and px < 7 and not side in BoardTemp[py+1, px+1]:
                    moves.append(((px,px),(px+1,py+1),Player[0]))
                if py > 0 and px < 7 and not side in BoardTemp[py-1, px+1]:
                    moves.append(((px,px),(px+1,py-1),Player[0]))
                if px < 7 and not side in BoardTemp[py, px+1]:
                    moves.append(((px,px),(px+1,py),Player[0]))
                if px > 0 and not side in BoardTemp[py, px-1]:
                    moves.append(((px,px),(px-1,py),Player[0]))
                if px > 0 and py > 0 and not side in BoardTemp[py-1, px-1]:
                    moves.append(((px,px),(px-1,py-1),Player[0]))
                if px > 0 and py < 7 and not side in BoardTemp[py+1, px-1]:
                    moves.append(((px,px),(px-1,py+1),Player[0]))
                if py < 7 and not side in BoardTemp[py-1, px]:
                    moves.append(((px,px),(px,py+1),Player[0]))
            elif(Player[0] == "Queen"):
                    px = Player[1][1]
                    py = Player[1][0]
                    while px < 7 and not side in BoardTemp[py, px+1]:
                        moves.append(((px,px),(px+1,py),Player[0]))
                        px = px+1
                        if opposide in BoardTemp[py,px]:
                            break
                    px = Player[1][1]
                    py = Player[1][0]
                    while px > 0 and not side in BoardTemp[py, px-1]:
                        moves.append(((px,px),(px-1,py),Player[0]))
                        px = px-1
                        if opposide in BoardTemp[py,px]:
                            break
                    px = Player[1][1]
                    py = Player[1][0]
                    while py > 0 and not side in BoardTemp[py-1, px]:
                        moves.append(((px,px),(px,py-1),Player[0]))
                        py = py-1
                        if opposide in BoardTemp[py,px]:
                            break
                    px = Player[1][1]
                    py = Player[1][0]
                    while py < 7 and not side in BoardTemp[py+1, px]:
                        moves.append(((px,px),(px,py+1),Player[0]))
                        py = py+1
                        px = Player[1][1]
                        if opposide in BoardTemp[py,px]:
                            break
                    px = Player[1][1]
                    py = Player[1][0]
                    while px+1 <= 7 and py+1 <= 7 and not side in BoardTemp[py+1, px+1]:
                        moves.append(((px,px),(px+1,py+1),Player[0]))
                        px = px+1
                        py = py+1
                        if opposide in BoardTemp[py,px]:
                            break
                    px = Player[1][1]
                    py = Player[1][0]
                    while px-1 >= 0 and py+1 <= 7 and not side in BoardTemp[py+1, px-1]:
                            moves.append(((px,px),(px-1,py+1),Player[0]))
                            px = px-1
                            py = py+1
                            if opposide in BoardTemp[py,px]:
                               break
                    px = Player[1][1]
                    py = Player[1][0]
                    while px+1 <= 7 and py-1 >= 0 and not side in BoardTemp[py, px]:
                            moves.append(((px,px),(px+1,py-1),Player[0]))
                            px = px+1
                            py = py-1
                            if opposide in BoardTemp[py,px]:
                                break
                    px = Player[1][1]
                    py = Player[1][0]
                    while px-1 >= 0 and py-1 >= 0 and side not in BoardTemp[py, px]:
                            moves.append(((px,px),(px-1,py-1),Player[0]))
                            px = px-1
                            py = py-1
                            if opposide in BoardTemp[py,px]:
                                break
        shuffle(moves)
        if moves:
            bestmove =  moves[0]
            lastscore = float("-inf")
            for Move in moves:
                TempBoard,score = move(side,Move[0][0],Move[0][1],Move[1][0],Move[1][1],Board)
                if score > lastscore:
                    
                    lastscore = score
                    movescore.append(((Move[0][0],Move[0][1]),(Move[1][0],Move[1][1]),
                                    lastscore-SimMove(opposide,TempBoard,depth,i)))
                
            maxscore = 0
            shuffle(movescore)
            if movescore:
                BestMove = movescore[0]
                for Move in movescore:
                    if(Move[2] > 0 and i == 1):
                        print("Yay moving from {} {} to {} {} would grant : {}pts\n".
                            format(Move[0][1], Move[0][0], Move[1][1],Move[1][0],Move[2]))
                    if Move[2] > BestMove[2]:
                        BestMove = Move
                #print("Best move for move {} is {} {}{}=>{}{}".format(i,board[Move[0][1],Move[0][0]],Move[0][1],Move[0][0],Move[1][0],Move[1][1],))
                if(i == 1):

                    pyautogui.click(x1+remap(BestMove[0][0], 0, 8,0,x2-x1)+RektSize/2,y1+remap(BestMove[0][1],0,8,0,y2-y1)+RektSize/2)
                    time.sleep(1)
                    pyautogui.click(x1+remap(BestMove[1][0], 0, 8,0,x2-x1)+RektSize/2,y1+remap(BestMove[1][1],0,8,0,y2-y1)+RektSize/2)
                    time.sleep(1)
                    return BestMove[1]
                return BestMove[2]
    return 0


if __name__ == "__main__":
    for i in range(8):
        board.append([])
        for j in range(8):
            board[i].append("")
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('1'):  # if key 'q' is pressed
                x1, y1 = pyautogui.position()
                print("(x,y) = ({},{})".format(x1, y1))
                break  # finishing the loop
        except:
            continue
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('2'):  # if key 'q' is pressed
                x2, y2 = pyautogui.position()
                print("(x,y) = ({},{})".format(x2, y2))
                break  # finishing the loop
        except:
            continue
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('w'):  # if key 'q' is pressed
                Side = "White"

                break  # finishing the loop
            if keyboard.is_pressed('b'):  # if key 'q' is pressed
                Side = "Black"
                break  # finishing the loop
        except:
            continue
    RektSize = (y2-y1)/8
    print("Side = {}".format(Side))
    LoadPieces()
    while 1:
        print("Reading board")
        ReadBoard(Side)
