import AftFore as aft

if __name__ == '__main__':
    
    t_learn = [0.0, 1.0]            # The range of the learning period [day].
    t_test  = [1.0, 2.0]            # The range of the testing period [day].
    Data    = './AftFore/Kobe.txt'  # The path of the date file
    
    aft.EstFore(Data,t_learn,t_test)