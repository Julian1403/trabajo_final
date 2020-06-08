#%% Importamos las bibliotecas necesarias
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import LinearFIR_2
import pywt
import scipy.io.wavfile as wavf
from scipy.signal import welch
from scipy.stats import mannwhitneyu
import os
import pandas as pd


#%%Cremos las funciones necesarias para el cumplimiento de los puntos planteados
def load_record(path):
    """
    Función que recibe la ruta donde se aloja la señal, la lee y devuelve la señal
    y la frecuencia de muestreo
    """
    data, fs = librosa.load(path)
    return data, fs
#%%Punto 2
def filter_signal(data, fs, freqs):
    '''
    Función encargada de aplicar un filtro pasabandas a la señal.
    freqs debe ser una tupla o lista -> [low, high]
    Devuelve la resta entre la señal original y la filtrada, de este modo 
    eliminamos los ciclos cardiacos
    '''
    #Diseñamos los filtros
    # order, highpass = filter_design(fs, locutoff = freqs[0], hicutoff = 0, revfilt = 1)
    # order, lowpass = filter_design(fs, locutoff = 0, hicutoff = freqs[1], revfilt = 0)
    # #Aplicamos el filtro
    # print(highpass.shape)
    # data = signal.filtfilt(highpass, 1, data.T)
    # data= signal.filtfilt(lowpass, 1, data)
    # data = np.asfortranarray(data)
    data_f = LinearFIR_2.eegfiltnew(data, fs, 0, freqs[1], 0)
    data_f = LinearFIR_2.eegfiltnew(data_f, fs, freqs[0], 0, 1)
    #Retornamos loa señal sin los ciclos cardiacos
    time = np.arange(0, data.shape[0]/fs, 1/fs)
    return np.float32(data-data_f), time
#%%Punto 3
def filter_wavelet(data, attenuation="db6"):
    '''
    Función que filtra la señal usando el método de wavelet.
    Devuelve la resta entre la señal original y la filtrada, de este modo 
    eliminamos los ciclos cardiacos
    '''
    LL = int(np.floor(np.log2(data.shape[0])))
    coeff = pywt.wavedec( data, attenuation, level=LL )
    thr = wavelet.thselect(coeff)
    coeff_t = wavelet.wthresh(coeff,thr)
    x_rec = pywt.waverec( coeff_t, attenuation)
    x_rec = x_rec[0:data.shape[0]]
    #Retornamos loa señal sin los ciclos cardiacos
    return np.float32(data-x_rec)

#%%Punto 5 y 6
def strange_cycles(data, fs, path_annotations):
    '''
    Funcion encargada de extraer del archivo de anotaciones los inicios
    y finales de los ciclos respiratorios y con base en ellos extrae los 
    ciclos y clasifica cada ciclo en 3 categorías:
    0: Sanos
    1: Crepitancias
    2: Sibilancias
    3: Crepitancias y Sibilancias
    Además calcula la varianza, el rango, elpromedio de media movil y el 
    promedio de la densidad de poder espectral (PSD) de cada ciclo.

    Devuelve una lista con la siguiente estructura:
    [clasificación, varianza, rango, mediaMovil, mediaPSD]
    '''
    with open(path_annotations, 'r') as file:
        lines = file.readlines()
    file.close()
    time = np.arange(0, (data.shape[0]+1)/fs, 1/fs)
    cycles = []
    for i in lines:
        dic = dict()
        datos = i[:-1].split("\t")
        inicio =  np.where(time >= float(datos[0]))[0]
        fin =  np.where(time >= float(datos[1]))[0]
        cycle = data[inicio[0]:fin[0]]
        if datos[2] == "0" and datos[3] == "0": clasification = 0
        elif datos[2] == "1" and datos[3] == "0": clasification = 1
        elif datos[2] == "0" and datos[3] == "1": clasification = 2
        else: clasification = 3
        dic["clasification"] = clasification
        #Hasta aquí punto 5
        #De aquí es punto 6
        dic["var"] = cycle.var()
        dic["range"] = cycle.max() - cycle.min()
        dic["ssm"] = moving_average(cycle, 100)
        dic["mean_welch"] = p_welch(cycle, fs)
        cycles.append(dic)
    return cycles    

def moving_average(data, window):
    '''
    Función que extrae calcula el promedio de media movil
    '''
    m_a =  np.convolve(data, np.ones(window), 'valid') / window
    return m_a.mean()

def p_welch(data, fs):
    '''
    Función que calcula el periodograma de welch y retorna su promedio
    '''
    f, pxx = welch(data, fs, "hamming")
    return pxx.mean()


def create_dataframe(path, name_dataframe):
    files = os.listdir(path)
    datos = []
    for i in range(0, len(files), 2):
        data, fs = load_record(os.path.join(path,files[i+1]))
        f_w = filter_wavelet(data, "db36")
        cycles = strange_cycles(data, fs, os.path.join(path,files[i]))
        datos.extend(cycles)
    return pd.DataFrame(datos)
    
def histogramas(path):
    '''
    Funcion que carga un dataframe y a partir con la siguiente estructura
    clasificacion   varianza    rango   mediaMovil  mediaPSD
    y a partir de esta grafica el histograma de cada columna dependiendo 
    de la clasificacion de los datos 
    0: Sanos
    1: Crepitancias
    2: Sibilancias
    3: Crepitancias y Sibilancias
    Además calcula la U Mann whitney de cada una de las columnas entre los
    ciclos clasificados como sanos y las demás clasificaciones.

    Retorna un array que contiene los resultados de la U Mann whitney
    '''
    df = pd.read_csv(path)
    #Obtenemos la varianza, rango, promedio movil, promedio de potencia y graficamos el histograma
    #los ciclos sanos
    sanos = df.loc[:,"clasification"] == 0
    sanos = df.loc[sanos]

    sanos_var = sanos.loc[:,"var"]
    plt.figure(1)
    sanos_var.hist()
    plt.title("Varianza ciclos sanos")

    sanos_range = sanos.loc[:,"range"]
    plt.figure(2)
    sanos_range.hist()
    plt.title("Rango ciclos sanos")

    sanos_ssm = sanos.loc[:,"ssm"]
    plt.figure(3)
    sanos_ssm.hist()
    plt.title("Promedio móvil ciclos sanos")

    sanos_w = sanos.loc[:,"mean_welch"]
    plt.figure(4)
    sanos_w.hist()
    plt.title("Promedio PSD ciclos sanos")
    
    #Obtenemos la varianza, rango, promedio movil, promedio de potencia y graficamos el histograma
    #los ciclos con crepitancias
    crepitancias = df.loc[:,"clasification"] == 1
    crepitancias = df.loc[crepitancias]

    crepitancias_var = crepitancias.loc[:,"var"]
    plt.figure(5)
    crepitancias_var.hist()
    plt.title("Varianza ciclos con crepitancias")

    crepitancias_range = crepitancias.loc[:,"range"]
    plt.figure(6)
    crepitancias_range.hist()
    plt.title("Rango ciclos con crepitancias")

    crepitancias_ssm = crepitancias.loc[:,"ssm"]
    plt.figure(7)
    crepitancias_ssm.hist()
    plt.title("Promedio móvil ciclos con crepitancias")

    crepitancias_w = crepitancias.loc[:,"mean_welch"]
    plt.figure(8)
    crepitancias_w.hist()
    plt.title("Promedio PSD ciclos con crepitancias")

    #Obtenemos la varianza, rango, promedio movil, promedio de potencia y graficamos el histograma
    #los ciclos con sibilancias
    sibilancias = df.loc[:,"clasification"] == 2
    sibilancias = df.loc[sibilancias]

    sibilancias_var = sibilancias.loc[:,"var"]
    plt.figure(9)
    sibilancias_var.hist()
    plt.title("Varianza ciclos con sibilancias")

    sibilancias_range = sibilancias.loc[:,"range"]
    plt.figure(10)
    sibilancias_range.hist()
    plt.title("Rango ciclos con sibilancias")

    sibilancias_ssm = sibilancias.loc[:,"ssm"]
    plt.figure(11)
    sibilancias_ssm.hist()
    plt.title("Promedio móvil ciclos con sibilancias")

    sibilancias_w = sibilancias.loc[:,"mean_welch"]
    plt.figure(12)
    sibilancias_w.hist()
    plt.title("Promedio PSD ciclos con sibilancias")

    #Obtenemos la varianza, rango, promedio movil, promedio de potencia y graficamos el histograma
    #los ciclos con crepitancias y sibilancias
    crep_sib = df.loc[:,"clasification"] == 3
    crep_sib = df.loc[crep_sib]

    crep_sib_var = crep_sib.loc[:,"var"]
    plt.figure(13)
    crep_sib_var.hist()
    plt.title("Varianza ciclos con crepitancias y sibilancias")

    crep_sib_range = crep_sib.loc[:,"range"]
    plt.figure(14)
    crep_sib_range.hist()
    plt.title("Rango ciclos con crepitancias y sibilancias")

    crep_sib_ssm = crep_sib.loc[:,"ssm"]
    plt.figure(15)
    crep_sib_ssm.hist()
    plt.title("Promedio móvil ciclos con crepitancias y sibilancias")

    crep_sib_w = crep_sib.loc[:,"mean_welch"]
    plt.figure(16)
    crep_sib_w.hist()
    plt.title("Promedio PSD ciclos con crepitancias y sibilancias")
    plt.show()

    #Prueba de U Mann whitney
    resultados = np.ones((12, 2))
    
    resultados[0,:]  =  mannwhitneyu(sanos_var, crepitancias_var)
    resultados[1,:]  =  mannwhitneyu(sanos_range, crepitancias_range)
    resultados[2,:]  =  mannwhitneyu(sanos_ssm, crepitancias_ssm)
    resultados[3,:]  =  mannwhitneyu(sanos_w, crepitancias_w)
    resultados[4,:]  =  mannwhitneyu(sanos_var, sibilancias_var)
    resultados[5,:]  =  mannwhitneyu(sanos_range, sibilancias_range)
    resultados[6,:]  =  mannwhitneyu(sanos_ssm, sibilancias_ssm)
    resultados[7,:]  =  mannwhitneyu(sanos_w, sibilancias_w)
    resultados[8,:]  =  mannwhitneyu(sanos_var, crep_sib_var)
    resultados[9,:]  =  mannwhitneyu(sanos_range, crep_sib_range)
    resultados[10,:]  =  mannwhitneyu(sanos_ssm, crep_sib_ssm)
    resultados[11,:]  =  mannwhitneyu(sanos_w, crep_sib_w)
    return resultados

#%%Implementación punto 1
# data, fs = load_record(r'Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files\104_1b1_Ar_sc_Litt3200.wav')

#%%Implementación punto 2
# data1, time = filter_signal(data, fs, [500, 1000])

#%%Implementación punto 3
# f_w = filter_wavelet(data, "db36")

#%%Punto 4
# plt.figure(1)
# plt.plot(time, data)
# plt.plot(time, data1)

# plt.figure(2)
# plt.plot(time, data)
# plt.plot(time, f_w)
# plt.show()
#%%Guardamos los audios amplificando 100 veces para escuchar
#los sonidos
# wavf.write("Filtro_Fir.wav", fs, data1*100)
# wavf.write("Filtro_Wavelet.wav", fs, f_w*100)

#%%Implementación punto 5 y 6
# cycles = strange_cycles(data, fs, r'Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files\104_1b1_Ar_sc_Litt3200.txt')

#%%Implementación punto 7
# frame = create_dataframe(r"Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files", "HOOOOLi")
# print(frame.to_string())
# frame.to_csv("dataframe.csv")

#%%Mostramos los histogramas y calculamos la U Mann whitney 
resultados = histogramas("dataframe.csv")
print("U Mann whitney para varianza de sanos vs varianza de crepitancias:",resultados[0,:])
print("U Mann whitney para rango de sanos vs rango de crepitancias:",resultados[1,:])
print("U Mann whitney para promedio movil de sanos vs promedio movil de crepitancias:",resultados[2,:])
print("U Mann whitney para promedio PSD de sanos vs promedio PSD de crepitancias:",resultados[3,:])
print("U Mann whitney para varianza de sanos vs varianza de sibilancias:",resultados[4,:])
print("U Mann whitney para rango de sanos vs rango de sibilancias:",resultados[5,:])
print("U Mann whitney para promedio movil de sanos vs promedio movil de sibilancias:",resultados[6,:])
print("U Mann whitney para promedio PSD de sanos vs promedio PSD de sibilancias:",resultados[7,:])
print("U Mann whitney para varianza de sanos vs varianza de crepitancias y sibilancias:",resultados[8,:])
print("U Mann whitney para rango de sanos vs rango de crepitancias y sibilancias:",resultados[9,:])
print("U Mann whitney para promedio movil de sanos vs promedio movil de crepitancias y sibilancias:",resultados[10,:])
print("U Mann whitney para promedio PSD de sanos vs promedio PSD de crepitancias y sibilancias:",resultados[11,:])
