import random
from PIL import Image

def pil_loader(path):
    img = Image.open(path)
    bands = img.getbands()
    if len(bands) != 3:  # changed >3 to !=3 
        #print(len(bands)) 
        img = img.convert('RGB')
    return img

class Resampler(object):
    def __init__(self):
        pass

    @classmethod
    def resample(self, slices, threshold, offset=0):
        '''
        Args:
            slices: the list of slices that requires upsampling.
            threshold: the expected number of slices
        '''
        if threshold == len(slices):
            return slices
        elif threshold > len(slices):
            return self.upsample(slices, threshold)
        else:
            return self.undersample(slices, threshold, offset)

    @staticmethod
    def upsample(slices, threshold=64):
        raise NotImplementedError

    @staticmethod
    def undersample(slices, threshold=64, offset=0):
        raise NotImplementedError


class RandomResampler(Resampler):
    @staticmethod
    def upsample(slices, threshold=64):
        original_num = len(slices)
        d = threshold - original_num
        tmp = []
        idxs = []
        for _ in range(d):
            idx = random.randint(0, original_num-1)
            idxs.append(idx)
        for idx, value in enumerate(slices):
            tmp.append(value)
            while idx in idxs:
                idxs.remove(idx)
                tmp.append(value)
        return tmp

    @staticmethod
    def undersample(slices, threshold=64,offset=0):
        original_num = len(slices)
        d = original_num - threshold
        tmp = slices.copy()
        for _ in range(d):
            idx = random.randint(0, len(tmp)-1)
            tmp.pop(idx)
        return tmp

class SymmetricalResampler(Resampler):
    '''
    Examples:
        ```
        a = list(range(7))
        re = SymmetricalResampler()
        re.resample(a,15) # [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6]
        re.resample(a,10) # [0, 1, 1, 2, 3, 3, 4, 5, 5, 6]
        re.resample(a,3) # [1, 3, 5]

        ```
    '''
    @staticmethod
    def upsample(slices, threshold=64):
        tmp = []
        original_num = len(slices)
        add_num = threshold - original_num
        add_idxs = list(range(original_num))
        if add_num % original_num == 0:
            repetitions = add_num // original_num
        else:
            repetitions = add_num // original_num + 1

        if repetitions > 1:
            for _ in range(repetitions-1): add_idxs.extend(list(range(original_num)))

        remain_num = threshold - len(add_idxs)
        if remain_num == 0:
            return tmp
        else:
            interval = original_num // remain_num
            idx = original_num // 2 if original_num%2==1 else original_num//2 -1
            remain_list = [idx]
            count = 1
            flag = 'right'
            while len(remain_list) < remain_num:
                if flag == 'right':
                    idx = idx + count * interval
                    flag = 'left'
                elif flag == 'left':
                    idx = idx - count * interval
                    flag = 'right'
                count += 1
                remain_list.append(idx)
            add_idxs.extend(remain_list)
            for idx in sorted(add_idxs):
                tmp.append(slices[idx])
            return tmp

    @staticmethod
    def undersample(slices, threshold=64,offset=0):
        tmp = []
        original_num = len(slices)
        add_idxs = []
        remain_num = threshold
        interval = original_num // remain_num

        idx = original_num // 2 if original_num%2==1 else original_num//2 -1 
        idx += offset 
        remain_list = [idx]
        count = 1
        flag = 'right'

        #print("idx = {}".format(idx))  

        while len(remain_list) < remain_num:
            if flag == 'right':
                idx = idx + count * interval
                flag = 'left'
            elif flag == 'left':
                idx = idx - count * interval
                flag = 'right'
            count += 1 
            remain_list.append(idx)
            #print("idx = {}".format(idx))  

        add_idxs.extend(remain_list)
        for idx in sorted(add_idxs):
            tmp.append(slices[idx])
        return tmp


def SymmetricalSequentialResampler(slices_all, length, center=True):

    #print("slices_all = {}".format(slices_all))   
    slices_num = len(slices_all) 
    interval = int(slices_num // length)
    #print(len(slices_all),interval) 

    slices_len = 0  
    slices_total = []        
    slices_to_process = dict()  
    i = 0 
    num_sets = []
 
    debug = False
 
    count_sets = 0 
    for i in range(interval): 
        #print("i = {}, interval = {}".format(i,interval))                
        slices = SymmetricalResampler.resample(slices_all, length, i-interval+1)
        slices_len += len(slices)
        slices_total.extend(slices)
        #print("i = {}, len= {}, slices = {}".format(i,slices_len,slices))  
        slices_to_process[count_sets] = slices 
        if debug: 
            print("Type 1.1, count_sets = {}, sets = {}".format(count_sets,slices))
        count_sets +=1 
    num_sets.append(count_sets)  
                                   
    if slices_num - len(slices_total) > 0 and interval==0:  
        i = interval
        slices_remain = set(slices_all) - set(slices_total)
        slices_remain = list(slices_remain) 
        slices_remain.sort(key=lambda x: int(x.split('.')[0])) 
        slices = SymmetricalResampler.resample(slices_remain, length) 
        slices_to_process[count_sets] = slices
        if debug: 
            print("Type 1.2, count_sets = {}, sets = {}".format(count_sets,slices))
        count_sets +=1 
    num_sets.append(count_sets) 
            
    #06/26 added sequential slices 

    #get one set at the center 
    if slices_num > length and center:  
        offset = (slices_num - length)//2 
        slices = slices_all[offset:offset+length]
        slices_to_process[count_sets] = slices
        if debug: 
            print("Type 2.1, count_sets = {}, sets = {}".format(count_sets,slices))

        count_sets +=1 
        num_sets.append(count_sets)  

    if slices_num > length and not center: 
        i +=1
        if slices_num == length * interval:  
            avg_len = int(slices_num/interval)
        else: 
            avg_len = int(slices_num // (interval+1))
        #print(slices_len, interval,avg_len) 
        slices_total = []   
        for j in range(interval):
            slices = slices_all[j*avg_len:(j+1)*avg_len]       
            slices = SymmetricalResampler.resample(slices, length)  
            slices_to_process[count_sets] = slices
            slices_total.extend(slices)
            if debug: 
                print("Type 3.1, count_sets = {}, sets = {}".format(count_sets,slices))
            count_sets +=1 
        num_sets.append(count_sets)  

        if slices_num - len(slices_total) > 0: 
            j +=1 
            slices_remain = set(slices_all) - set(slices_total)
            slices_remain = list(slices_remain) 
            slices_remain.sort(key=lambda x: int(x.split('.')[0])) 
            slices = SymmetricalResampler.resample(slices_remain, length)   
            slices_to_process[count_sets] = slices
            if debug: 
                print("Type 3.2, count_sets = {}, sets = {}".format(count_sets,slices))
            count_sets +=1 
        num_sets.append(count_sets)  

        #print(slices_to_process)  
        #input("dbg")    

    return slices_to_process, num_sets 


