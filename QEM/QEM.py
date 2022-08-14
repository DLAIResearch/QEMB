import numpy as np
import cv2 as cv
import math,cmath
import torch
import torchvision
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def EM_D(img, maxorder):
    alpha = 1;
    N=np.size(img[0,:])
    M=10*N
    G=np.zeros(shape=(M,M))
    for u in range(M):
        for v in range(M):
            rho=np.power(u / M, 1 / alpha)
            theta = (2 * np.pi * (v)) / M
            k = int(np.ceil(rho * (N / 2) * np.sin(theta)))
            l = int(np.ceil(rho * (N / 2) * np.cos(theta)))
            a=img[int((-1) * k + (N / 2) ), int(l + (N / 2))-1]
            b=np.sqrt((np.power((u+1) / M,  (2 / alpha - 1)) / (2 * np.pi * alpha)))
            G[u,v] = a * b
    TEMP = np.fft.fft2(G)
    TEMP =np.sqrt(2 * np.pi) * TEMP / (np.power(M,2))
    output = np.zeros(shape=(2 * maxorder + 1, 2 * maxorder + 1),dtype='complex')
    output[0: maxorder, 0: maxorder] = TEMP[M - maxorder : M, M - maxorder : M]
    output[0: maxorder, maxorder : 2 * maxorder+1] = TEMP[M - maxorder: M, 0: maxorder+1]
    output[maxorder: 2 * maxorder+1, 0: maxorder] = TEMP[0: maxorder+1, M - maxorder: M]
    output[maxorder: 2 * maxorder+1, maxorder : 2 * maxorder+1] = TEMP[0: maxorder+1, 0: maxorder+1]
    return output
def EM_Dm(img, maxorder):
    alpha = 1;
    N=np.size(img[0,:])
    M=10*N
    G=np.zeros(shape=(M,M))
    for u in range(M):
        for v in range(M):
            rho=np.power(u / M, 1 / alpha)
            theta = (2 * np.pi * (v)) / M
            k = int(np.ceil(rho * (N / 2) * np.sin(theta)))
            l = int(np.ceil(rho * (N / 2) * np.cos(theta)))
            a=img[int((-1) * k + (N / 2) ), int(l + (N / 2))-1]
            b=np.sqrt((np.power((u+1) / M,  (2 / alpha - 1)) / (2 * np.pi * alpha)))
            G[u,v] = a * b
    TEMP = np.fft.fft2(G)
    TEMP =np.sqrt(2 * np.pi) * TEMP / (np.power(M,2))
    output = np.zeros(shape=(2 * maxorder + 1, 2 * maxorder + 1),dtype='complex')
    output[0: maxorder, 0: maxorder] = TEMP[M - maxorder : M, M - maxorder : M]
    output[0: maxorder, maxorder : 2 * maxorder+1] = TEMP[M - maxorder: M, 0: maxorder+1]
    output[maxorder: 2 * maxorder+1, 0: maxorder] = TEMP[0: maxorder+1, M - maxorder: M]
    output[maxorder: 2 * maxorder+1, maxorder : 2 * maxorder+1] = TEMP[0: maxorder+1, 0: maxorder+1]
    det = 0.5
    #Binary image
    se = np.array([[0, 1, 0, 1, 0, 1, 1, 0],
                   [0, 1, 0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0, 0, 0],
                   [0, 1, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 1, 1, 0],
                   [0, 1, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0, 1]])
    se_size = np.size(se[1])
    ex_size = np.size(output[1])
    for u in range(se_size):
        for v in range(se_size):
            p=  int(((ex_size +1) / 2) - u )
            q=  int(((ex_size + 1) / 2) + v )
            if se[u, v] == 0:
                output[p, q] = (output[p, q] - det) * (output[p, q] / output[p, q])

            else:
                output[p, q] = (output[p, q] + det) * (output[p, q] / output[p, q])

    return output

def EM_R(img,moments,maxorder,mag,ang):
    N = M = img
    r=mag
    th=ang
    pz=th<0
    theta = np.zeros(shape=(N, M))
    theta[pz] = th[pz] + 2 * np.pi
    theta[~pz] = th[~pz]
    pz =r > 1
    rho = np.zeros(shape=(N, M))
    rho[pz] = 0.5
    rho[~pz] = r[~pz]
    output =np.zeros(shape=(N, M))
    for u in range(2*maxorder+1):
        order = -maxorder + u
        R = np.multiply(np.sqrt(rho** (-1)), np.exp(np.multiply(1j * 2 * np.pi * order,rho)))
        for v in range( 2 * maxorder+1):
            repetition = -maxorder + v
            moment = moments[u, v]
            pupil = np.multiply(R,np.exp(1j * repetition * theta))
            output = output + moment * pupil
            output[r > 1] = 0
    return output

def EM_TD(R_fft,G_fft,B_fft):

    ansA = -np.sqrt(1 / 3) * (R_fft.imag + G_fft.imag + B_fft.imag)
    ansB = R_fft.real + np.sqrt(1 / 3) * (G_fft.imag - B_fft.imag)
    ansC = G_fft.real + np.sqrt(1 / 3) * (B_fft.imag - R_fft.imag)
    ansD = B_fft.real + np.sqrt(1 / 3) * (R_fft.imag - G_fft.imag)
    return ansA, ansB, ansC, ansD
def embedding(ansA, ansB, ansC, ansD):
    det = 0.75
    # se =np.random.randint(0, 2, (8, 8))
    #  Binary images
    se = np.array([[0, 1, 0, 1, 0, 1, 1, 0],
                    [0, 1, 0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 1],
                    [0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 0],
                    [0, 1, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 1]])
    se_size=np.size(se[1])
    ex_size=np.size(ansA[1])
    A = np.sqrt(ansA ** 2 + ansB ** 2 + ansC ** 2 + ansD ** 2)
    for u in range(se_size):
        for v in range(se_size):
        #Low-order moment perturbations
         p=  int(((ex_size +1) / 2) - u )
         q=  int(((ex_size +1) / 2) + v )
        #High-order moment perturbations
         #p=int(u)
         #q=int(v)
         if se[u,v]==0:
            ansA[p,q] = (A[p,q] - det) * (ansA[p,q] / A[p,q])
            ansB[p,q] = (A[p,q] - det) * (ansB[p,q] / A[p,q])
            ansC[p,q] = (A[p,q] - det) * (ansC[p,q] / A[p,q])
            ansD[p,q] = (A[p,q] - det) * (ansD[p,q] / A[p,q])
         else:
            ansA[p,q] = (A[p,q] + det) * (ansA[p,q] / A[p,q])
            ansB[p,q] = (A[p,q] + det) * (ansB[p,q] / A[p,q])
            ansC[p,q] = (A[p,q] + det) * (ansC[p,q] / A[p,q])
            ansD[p,q] = (A[p,q] + det) * (ansD[p,q] / A[p,q])

    return ansA, ansB, ansC, ansD

def EM_TR(image,ansA,ansR,ansG,ansB,maxorder,mag,ang):

    anmA=EM_R(image,ansA,maxorder,mag, ang)
    anmB=EM_R(image,ansR,maxorder,mag, ang)
    anmC=EM_R(image,ansG,maxorder,mag, ang)
    anmD=EM_R(image,ansB,maxorder,mag, ang)
    return anmA, anmB, anmC, anmD

def EM_RImage(anmA,anmB,anmC,anmD):

    fA = anmA.real-np.sqrt(1 / 3) *(anmB.imag+anmC.imag+anmD.imag)
    fB = anmB.real + np.sqrt(1 / 3) * (anmA.imag+anmC.imag-anmD.imag)
    fC = anmC.real + np.sqrt(1 / 3) * (anmA.imag-anmB.imag+anmD.imag)
    fD = anmD.real + np.sqrt(1 / 3) * (anmA.imag+anmB.imag-anmC.imag)

    return fA,fB,fC,fD

#Three-channel poisoned image generation
def QEM(single_image1,order):
    torchvision.utils.save_image(single_image1, './original_image.JPEG')
    single_image1=single_image1.cpu().numpy()
    single_image1=single_image1.transpose(1, 2, 0)
    single_image = copy.deepcopy(single_image1)
    single_image=single_image*255
    original=copy.deepcopy(single_image1)
    original1 = copy.deepcopy(single_image1)
    red = single_image[:, :, 0]
    green = single_image[:, :, 1]
    blue = single_image[:, :, 2]
    N = np.size(red[1])  # 图像大小
    N = int(N)
    x = np.linspace(-1 + (1 / N), 1 - (1 / N), np.size(red[1, :]))
    y = np.linspace(-1 + (1 / N), 1 - (1 / N), np.size(red[1, :]))
    xx, yy = np.meshgrid(x, y, indexing='xy')
    mag, ang = cv.cartToPolar(xx, yy, angleInDegrees=False)
    ang[0:int(N / 2), :] = np.pi * 2 - ang[0:int(N / 2), :]
    ang[int(N / 2):, :] = -ang[int(N / 2):, :]
    green[mag > 1] = 0
    red[mag > 1] = 0
    blue[mag > 1] = 0
    moments_R = EM_D(red, order)
    moments_G = EM_D(green, order)
    moments_B = EM_D(blue, order)

    ansA, ansB, ansC, ansD = EM_TD(moments_R, moments_G, moments_B)
    ansA1, ansB1, ansC1, ansD1 = embedding(ansA, ansB, ansC, ansD)
    anmA1, anmB1, anmC1, anmD1 = EM_TR(N, ansA1, ansB1, ansC1, ansD1, order,mag,ang)

    red1 = original[:, :, 0]
    green1 = original[:, :, 1]
    blue1 = original[:, :, 2]

    green1[mag > 0.9722] = 0
    red1[mag >0.9722]= 0
    blue1[mag >0.9722] = 0

    green1[0.05>mag] = 0
    red1[0.05>mag]= 0
    blue1[0.05>mag] = 0

    part1=original1-np.transpose(np.array([red1, green1, blue1]), (1, 2, 0))
    fA1, fB1, fC1, fD1 = EM_RImage(anmA1, anmB1, anmC1, anmD1)
    fB1[mag > 0.9722] = 0
    fC1[mag > 0.9722]= 0
    fD1[mag> 0.9722] = 0
    fB1[0.05>mag] = 0
    fC1[0.05>mag]= 0
    fD1[0.05>mag] = 0
    part2_2 = np.transpose(np.array([np.abs(fB1), np.abs(fC1), np.abs(fD1)]), (1, 2, 0))
    part2_2 = part2_2 / 255
    image_s=(part1+part2_2).transpose(2,0,1)
    image_s=torch.from_numpy(image_s).to(device)
    image_s=torch.clamp(image_s, 0, 1)
    torchvision.utils.save_image(image_s, './back_image.JPEG')
    return image_s
#Single-channel poisoned image generation
def EM(single_image1,order):
    torchvision.utils.save_image(single_image1, './original_image.JPEG')
    single_image1=single_image1.cpu().numpy()
    single_image1=single_image1.transpose(1, 2, 0)
    single_image = copy.deepcopy(single_image1)
    single_image=single_image*255
    original=copy.deepcopy(single_image1)
    original1 = copy.deepcopy(single_image1)
    single = single_image[:, :, 0]
    N = int(np.size(single[1])) # 图像大小
    x = np.linspace(-1 + (1 / N), 1 - (1 / N), np.size(single[1, :]))  # 横坐标
    y = np.linspace(-1 + (1 / N), 1 - (1 / N), np.size(single[1, :]))  # 纵坐标
    xx, yy = np.meshgrid(x, y, indexing='xy')  # xx从负一到1(0,N)yy(N,0)
    mag, ang = cv.cartToPolar(xx, yy, angleInDegrees=False)
    ang[0:int(N / 2), :] = np.pi * 2 - ang[0:int(N / 2), :]
    ang[int(N / 2):, :] = -ang[int(N / 2):, :]
    single[mag > 1] = 0
    moments_R = EM_D(single, order)
    single1 = original[:, :, 0]
    single1[mag >0.9922]= 0
    single1[mag < 0.05] = 0

    part1=original1[:, :, 0]-single1
    fA1 = EM_R(N,moments_R,order,mag,ang)
    fA1[mag > 0.9722] = 0
    fA1[mag < 0.05] = 0
    part2_2 = np.array(np.abs(fA1))
    part2_2 = part2_2 / 255
    image_s=(part1+part2_2)
    image_s=torch.from_numpy(image_s).unsqueeze(0).to(device)
    image_s=torch.clamp(image_s, 0, 1)
    torchvision.utils.save_image(image_s, './back_image1.JPEG')
    return image_s