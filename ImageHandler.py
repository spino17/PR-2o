import numpy as np

class ImageHandler:
    
    img = np.array([[], [], []])
    
    def __init__(self, img):
        self.img = img
        
    def ToPatches(self):
        patchvec = []
        row = 0
        col = 0
        height, widht = self.img.shape[:2]
        num_row = int(height / 32)
        num_col = int(widht / 32)
        
        for col in range(0, num_col):
            for row in range(0, num_row):
                bias_i = col * 32
                bias_j = row * 32
                red = [0, 0, 0, 0, 0, 0, 0, 0]
                green = [0, 0, 0, 0, 0, 0, 0, 0]
                blue = [0, 0 ,0 , 0, 0, 0, 0, 0]
                for i in range(0, 32):
                    for j in range(0, 32):
                        index_r = int(8 * self.img[bias_j + j][bias_i + i][0] / 255)
                        index_g = int(8 * self.img[bias_j + j][bias_i + i][1] / 255)
                        index_b = int(8 * self.img[bias_j + j][bias_i + i][2] / 255)
                        if (index_r == 8):
                            index_r = 7
                        if (index_g == 8):
                            index_g = 7
                        if (index_b == 8):
                            index_b = 7
                        
                        red[index_r] = red[index_r] + 1
                        green[index_g] = green[index_g] + 1
                        blue[index_b] = blue[index_b] + 1
                
                patch = red + green + blue
                patchvec.append(patch)
        
        max_row = num_row
        max_col = num_col
        
        col_end = widht - max_col * 32
        row_end = height - max_row * 32
        
        
        for row in range(0, num_row):
            bias_i = max_col * 32
            bias_j = row * 32
            red = [0, 0, 0, 0, 0, 0, 0, 0]
            green = [0, 0, 0, 0, 0, 0, 0, 0]
            blue = [0, 0 ,0 , 0, 0, 0, 0, 0]
            for i in range(0, col_end):
                for j in range(0, 32):
                    index_r = int(8 * self.img[bias_j + j][bias_i + i][0] / 255)
                    index_g = int(8 * self.img[bias_j + j][bias_i + i][1] / 255)
                    index_b = int(8 * self.img[bias_j + j][bias_i + i][2] / 255)
                    if (index_r == 8):
                        index_r = 7
                    if (index_g == 8):
                        index_g = 7
                    if (index_b == 8):
                        index_b = 7
                        
                    red[index_r] = red[index_r] + int(32 / col_end)
                    green[index_g] = green[index_g] + int(32 / col_end)
                    blue[index_b] = blue[index_b] + int(32 / col_end)
            if (col_end != 0):
                patch = red + green + blue
                patchvec.append(patch)
        
        for col in range(0, num_col):
            bias_i = col * 32
            bias_j = max_row * 32
            red = [0, 0, 0, 0, 0, 0, 0, 0]
            green = [0, 0, 0, 0, 0, 0, 0, 0]
            blue = [0, 0 ,0 , 0, 0, 0, 0, 0]
            for i in range(0, 32):
                for j in range(0, row_end):
                    index_r = int(8 * self.img[bias_j + j][bias_i + i][0] / 255)
                    index_g = int(8 * self.img[bias_j + j][bias_i + i][1] / 255)
                    index_b = int(8 * self.img[bias_j + j][bias_i + i][2] / 255)
                    if (index_r == 8):
                        index_r = 7
                    if (index_g == 8):
                        index_g = 7
                    if (index_b == 8):
                        index_b = 7
                        
                    red[index_r] = red[index_r] + int(32 / row_end)
                    green[index_g] = green[index_g] + int(32 / row_end)
                    blue[index_b] = blue[index_b] + int(32 / row_end)
            if (row_end != 0):
                patch = red + green + blue
                patchvec.append(patch)
     
        return patchvec
    

        
