# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:11:39 2018

@author: Ljx
"""
import matplotlib.pyplot as plt
import numpy as np, torch as t
import math

class Visualizer():
    def drawTest(self, iot, ts, drawLot = False):
        # i: T * batch * multi; t: future * batch * output_size; o: future * batch * output_size
        input_data, output_data , target_data = iot[0], iot[1], iot[2]
# =============================================================================
#         if input_data.shape[2] == 5:
#             batch = 0
#             for i in range(5):
#                 plt.subplot(1, 5, i+1)
#                 if not drawLot:
#                     plt.plot(ts[0][batch,...].cpu().numpy(), input_data[:, batch, i])
#                     plt.legend([])
#                 plt.plot(ts[1][batch,...].cpu().numpy(), output_data[:, batch, i])
#                 plt.plot(ts[1][batch,...].cpu().numpy(), target_data[:, batch, i])
#                 if not drawLot:
#                     plt.legend(['input', 'output','target'])
#                 else:
#                     plt.legend(['output','target'])
#             plt.show()
#             return
# =============================================================================
        
        batch = 0
        rows = 2
        columns = math.ceil(target_data.shape[2] * 1.0 / rows)
        print('time: ', ts[1][batch,0].item(), ' - ', ts[1][batch,-1].item())
        for i in range(target_data.shape[2]):
            plt.subplot(rows, columns, i+1)
#            print(ts[0][batch,...].numpy(), input_data[..., batch, i])
            if not drawLot:
                plt.plot(ts[0][batch,...].cpu().numpy(), input_data[:, batch, i])
                plt.legend([])
            print(output_data.shape)
            plt.plot(ts[1][batch,...].cpu().numpy(), output_data[:, batch, i])
            plt.plot(ts[1][batch,...].cpu().numpy(), target_data[:, batch, i])
            if not drawLot:
                plt.legend(['input', 'output','target'])
            else:
                plt.legend(['output','target'])
        
        plt.show()

    def drawEpochLoss(self, loss):
        plt.plot(loss)
        plt.show()