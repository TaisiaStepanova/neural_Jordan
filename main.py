from lib import *

if __name__ == '__main__':
    print("----------MENU----------")
    print("1) Training of neural network\n2) Prediction numbers\n")
    menu = input("Enter number: ")

    if menu == '1':
        print("1) Enter settings\n2) Default settings\n")
        sett = input("Enter number: ")

        p, e, alpha, N, col = 0, 0, 0, 0, 0
        if sett == '1':
            p = input("Enter p: ")
            e = input("Enter e: ")
            alpha = input("Enter alpha: ")
            N = input("Enter iteration: ")
            col = input("Enter col: ")
        elif sett == '2':
            p = 8
            e = 0.000001
            alpha = 0.1
            N = 100000
            col = 4

        train = input("Enter sequence number: ")

        training(p, e, alpha, N, col, int(train)-1)

    elif menu == '2':

        print('Enter the elements of the sequence, press "enter"')
        print('to finish typing press "enter"')
        a = float(input('-->> '))
        sequence = []
        while True:
            try:
                sequence.append([a])
                a = float(input('-->> '))
            except:
                break

        n = input("Enter the number of predictions: ")
        prediction(sequence, int(n))
