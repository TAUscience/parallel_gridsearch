import multiprocess
import time

from level_loads import level_loads as lloads
from knn import combinations as knn_comb
from knn import evaluate_knn as eknn
from random_forest import combinations_rf as rf_comb
from random_forest import evaluate_rf as erf
import data.load_data as ld

MODEL = 1
N_THREADS = 4
threads = []

X_train, X_test, y_train, y_test = ld.load_data()

if (MODEL == 1):
    splits = lloads(knn_comb, N_THREADS)
    lock = multiprocess.Lock()

    if __name__ == "__main__":
        for i in range(N_THREADS):
            threads.append(multiprocess.Process(target = eknn, args=(lock, splits[i], X_train, X_test, y_train, y_test)))
        
        start_time = time.perf_counter()
        
        # Se lanzan a ejecución
        for thread in threads:
            thread.start()
        # y se espera a que todos terminen
        for thread in threads:
            thread.join()
                
        finish_time = time.perf_counter()
        print(f"Program finished in {finish_time-start_time} seconds")

elif (MODEL == 2):
    splits = lloads(rf_comb, N_THREADS) 
    lock = multiprocess.Lock()

    if __name__ == "__main__":
        for i in range(N_THREADS):
            threads.append(multiprocess.Process(target = erf, args=(lock, splits[i], X_train, X_test, y_train, y_test)))
        
        start_time = time.perf_counter()
        
        # Lanzar los procesos
        for thread in threads:
            thread.start()
        # Esperar a que todos los procesos terminen
        for thread in threads:
            thread.join()

        finish_time = time.perf_counter()
        print(f"Program finished in {finish_time - start_time:.2f} seconds")
else:
    print("Modelo no válido")
