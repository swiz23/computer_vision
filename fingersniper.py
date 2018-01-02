import sys, pygame
from pygame.locals import *
import numpy as np
from matplotlib import pyplot as plt
import cv2

def make_video(screen):
    _image_num = 0
    while True:
        _image_num += 1
        str_num = "000" + str(_image_num)
        file_name = "image" + str_num[-4:] + ".jpg"
        pygame.image.save(screen, file_name)
        yield
kernel = np.ones((5,5),np.uint8)
cap = cv2.VideoCapture(0)
pygame.init()
pygame.display.set_caption("Finger Sniper")

size = width, height = 1336,1002 #640, 480
speed = [5, 5]
white = (255, 255, 255)
red = (255,0,0)
green = (0,255,0)
xf,yf = None,None

screen = pygame.display.set_mode(size)
screen.fill((200, 100, 50))
save_screen = make_video(screen)  # initiate the video generator
video = False  # at start: video not active

ball = pygame.image.load("redot.png")
ball = pygame.transform.scale(ball,(300,300))
ballrect = ball.get_rect()
bsize = ball.get_size()

target = pygame.image.load("target.png")
target = pygame.transform.scale(target,(50,50))
targetrect = target.get_rect()

myFont = pygame.font.SysFont("Times New Roman",32)
count = 0
astat = 'NOT ARMED'

stop_rect = False
avgcheck = []
avg = 0
counthit = 0

st = 5
game_done = False
calib_done = False

while cap.isOpened():
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                game_done = True
                break
            elif event.key == K_b:
                calib_done = True
                break
            elif event.key == pygame.K_v:
                # toggle video on/off by clicking 'v' on keyboard #
                video = not video


    ret, frame = cap.read()
    row,col,_ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cxb,cyb = ball.get_size()
    if game_done == True or cxb == 50:
        break

    if calib_done == True:
        calib_done = False
        mask = np.zeros((row,col), np.uint8)
        mask[int(row/2.-50):int(row/2.+50),int(col/2.-50):int(col/2.+50)] = 255
        masked_img = cv2.bitwise_and(frame,frame,mask = mask)

        # cv2.imshow('maskimg',cv2.flip(masked_img,1))
        # cv2.waitKey(1)
        masked_img = masked_img[int(row/2.-50):int(row/2.+50),int(col/2.-50):int(col/2.+50)]

        stop_rect = True
        mask_hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        hist_maskH = cv2.calcHist([mask_hsv],[0],None,[180],[0,180])
        hist_maskS = cv2.calcHist([mask_hsv],[1],None,[256],[0,256])
        flipH = (hist_maskH[::-1].copy())
        flipS = (hist_maskS[::-1].copy())
        index = hist_maskH.argmax()
        start = 0
        end = 0
        if index >= 80:
            start = 80
        else:
            end = 80
        l_h = np.argmax(hist_maskH[start:]>max(hist_maskH)*.02) + start
        l_s = np.argmax(hist_maskS>max(hist_maskS)*.02)
        u_h = 180 - (np.argmax(flipH[end:]>max(flipH)*.02)+end)
        u_s = 255 - np.argmax(flipS>max(flipS)*.02)
        # print l_h,l_s,u_h,u_s
        # plt.figure('histogram')
        # plt.subplot(211)
        # plt.plot(hist_maskH,'r')
        # plt.subplot(212)
        # plt.plot(hist_maskS,'g')
        # plt.show()
        lower = np.array([l_h,l_s,0],np.uint8)
        upper = np.array([u_h,u_s,255],np.uint8)

    if stop_rect == False:
        cv2.rectangle(frame,(int(col/2.-50),int(row/2.-50)),(int(col/2.+50),int(row/2.+50)),(0,255,0),3)

    # show the skin in the image along with the mask
    if stop_rect == True:
        skinMask = cv2.inRange(hsv_frame,lower,upper)
        # cv2.imshow('skinMask',skinMask)
        # cv2.waitKey(1)
        blurredSkin = cv2.GaussianBlur(skinMask,(7,7),3,3)
        blurredSkin = cv2.erode(blurredSkin,kernel,iterations = 1)
        blurredSkin = cv2.dilate(blurredSkin,kernel,iterations = 2)
        # cv2.imshow('blurredSkin',blurredSkin)
        # cv2.waitKey(1)

        contours= cv2.findContours(blurredSkin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) > 0:
            #find largest contour in mask, use to compute minEnCircle
            c = max(contours, key = cv2.contourArea)
           # print c
            moments = cv2.moments(c)
            if moments['m00'] != 0:
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
            cv2.circle(frame,(int(cx),int(cy)),10,255,-1)
            cv2.drawContours(frame,c,-1, (0,255,0),3)

            hull1 = cv2.convexHull(c)
            cv2.drawContours(frame,[hull1],0,(255,0,0),2)
            hull = cv2.convexHull(c,returnPoints = False)
            defects = cv2.convexityDefects(c,hull)

            if defects is not None and len(defects) > 0:
                s = defects[:,0][:,0]
                x = np.array(c[s][:,0][:,0], dtype=np.float)
                y = np.array(c[s][:,0][:,1], dtype=np.float)


                for i in range(defects.shape[0]):
                    cnt = c
                    s1,e1,f1,d1 = defects[i,0]
                    start = tuple(cnt[s1][0])
                    end = tuple(cnt[e1][0])
                    far = tuple(cnt[f1][0])
                    cv2.line(frame,start,far,[255,255,0],2)
                    cv2.line(frame,far,end,[255,255,0],2)
                    cv2.circle(frame,far,5,[0,255,255],-1)
                xp = cv2.pow(cv2.subtract(x, cx), 2)
                yp = cv2.pow(cv2.subtract(y, cy), 2)
                dist = cv2.sqrt(cv2.add(xp, yp))

                dist_max_i = np.argmax(dist)
                dmax = dist[dist_max_i]
                b = np.copy(dist).tolist()
                b.sort()
                bc = np.copy(dist).tolist()
                dc = np.copy(dist).tolist()

                i = 0
                while 1:
                    if len(bc) == 1:
                        break
                    else:
                        if (dmax-max(bc)) > [20]:
                            break
                        else:
                            bc.remove(b[-1-i])
                    i += 1

                if len(bc) > 3:
                    avgcheck.append(float(max(bc)[0])/dmax[0])
                else:
                    avg = 0

                if len(avgcheck) > 10:
                    avg = np.average(avgcheck)
                    avgcheck = []

                #print avg
                if avg < 0.79 and avg > 0.7:
                    #print 'Shot'
                    counthit += 1
                else:
                    #print 'No Shot'
                    counthit = 0

                if avg > 0.79 and avg < 0.86:
                    astat = 'ARMED'
                else:
                    astat = 'NOT ARMED'

                dist_max_2 = dc.index(max(bc))

                if dist_max_i < len(s):
                    farthest_defect = s[dist_max_i]
                    farthest_point = tuple(c[farthest_defect][0])
                    cv2.circle(frame,farthest_point,5,[0,0,255],-1)

                    xf,yf = farthest_point

                    xf,yf = int(round(xf*height/float(row))),int(round(yf*width/float(col)))
                if dist_max_2 < len(s):
                    farthest_defect_2 = s[dist_max_2]
                    farthest_point_2 = tuple(c[farthest_defect_2][0])
                    cv2.circle(frame,farthest_point_2,5,[0,255,0],-1)

        ballrect = ballrect.move(speed)

        if ballrect.left < 0 or ballrect.right > width:
            speed[0] = -speed[0]
        elif ballrect.top < 0 or ballrect.bottom > height:
            speed[1] = -speed[1]

        if ballrect.left < 0 or ballrect.right > width or ballrect.top < 0 or ballrect.bottom > height:

            ballarr = pygame.PixelArray(ball)
            ballarr = pygame.surfarray.pixels3d(ball)
            r,g,b = ballarr[ballarr.nonzero()[0:2]][len(ballarr)/2]
            ball = pygame.image.load("redot.png")
            ball = pygame.transform.scale(ball,(bsize[0]-st,bsize[1]-st))
            ballarr = pygame.surfarray.pixels3d(ball)
            ballarr[ballarr.nonzero()[0:2]] = [r,g,b]
            del ballarr
            xb,yb = ballrect.centerx, ballrect.centery
            ballrect = ball.get_rect()
            ballrect.centerx, ballrect.centery = xb,yb
            st += 5

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    snapshot = pygame.surfarray.make_surface(frame)
    snap = snapshot.copy()
    snap = pygame.transform.scale(snap,(size[0],size[1]))

    if stop_rect == True and xf != None:
        targetrect.centerx,targetrect.centery = size[0]-xf,yf

    if xf != None and counthit == 1 and ballrect.collidepoint(size[0]-xf,yf) == True:
        ballarr = pygame.PixelArray(ball)
        ballarr = pygame.surfarray.pixels3d(ball)
        r,g,b = np.random.random(3)*255
        ballarr[ballarr.nonzero()[0:2]] = [int(r),int(g),int(b)]
        del ballarr
        count += 1

    alabel = 'Gun is ' + astat
    hlabel = 'Number of Hits: '+ str(count)
    handlabel = 'Dist Ratio (aim for ~0.82): %.2f' % avg
    hitLabel = myFont.render(hlabel,1,white)

    if astat == 'ARMED':
        armedLabel = myFont.render(alabel,1,green)
        handnum = myFont.render(handlabel,1,green)
    else:
        armedLabel = myFont.render(alabel,1,red)
        handnum = myFont.render(handlabel,1,red)

    screen.blit(snap,(0,0))
    if stop_rect == True:
        screen.blit(ball, ballrect)
        if xf != None:
            screen.blit(target,targetrect)
        screen.blit(hitLabel,(700,50))
        screen.blit(armedLabel,(200,50))
        screen.blit(handnum,(200,100))
    pygame.display.flip()
    xf,yf = None, None
    if video:
        next(save_screen)  # call the generator


print 'Game Over'
print 'You scored '+ str(count) + ' hits!'
sys.exit()
pygame.quit()
cap.release()
