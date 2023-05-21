from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import torchaudio
import os
import torch
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import torch.nn.functional as F

def index(request):
    return render(request, 'home.html')


@csrf_exempt
def cough(request):
    print(request.method)
    if request.method == "POST":
        print("lala: ", request.method)
        print("query: ", request.FILES.get('query',None).name)
        print("s1: ", request.FILES.get('s1',None))
        print("s2: ", request.FILES.get('s2',None))
        print("s3: ", request.FILES.get('s3',None))
        print("s4: ", request.FILES.get('s4',None))
        audio = []
        query = request.FILES.get('query', None).name
        s1 = request.FILES.get('s1', None).name
        audio.append(s1)
        s2 = request.FILES.get('s2', None).name
        audio.append(s2)
        s3 = request.FILES.get('s3', None).name
        audio.append(s3)
        s4 = request.FILES.get('s4', None).name
        audio.append(s4)
        support = {}
        labels = []
        for file in audio:
            label = file.split("_")[0]
            if label not in labels:
                labels.append(label)
            if label not in support:
                support[label] = []
            for same in audio:
                if label == same.split("_")[0]:
                    if same not in support[label]:
                        support[label].append(same)
        print(support)
        print(labels)
        model_path = "D:\\HoangNgan\\DoAn\\best_model.pt"
        model = torch.load(model_path)
        xq_emb = model.encoder(getData(query).reshape(1, 1, 51, 40))
        su_label = []
        su_pro = []
        for key, value in support.items():
            su_label.append(key)
            tmp = []
            check = 0
            for i in value:
                feature = getData(i).reshape(1, 1, 51, 40)
                emb = model.encoder.forward(feature).squeeze()
                check += emb
                tmp.append(emb.detach().numpy())
            tmp = np.array(tmp)
            pro = (tmp).mean(0)
            su_pro.append(pro)
        n_class, n_query = torch.tensor(su_pro).shape[0], 1
        dists = euclidean_dist(xq_emb.reshape(1, 48), torch.tensor(su_pro))
        n_class, n_query = torch.tensor(su_pro).shape[0], 1
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        print(log_p_y)
        g, y_hat = log_p_y.max(0)
        print(log_p_y.max(0))
        print(g, y_hat)
        predict = labels[y_hat.squeeze()]

        # return redirect(request.path_info, {'predict': predict})
        return JsonResponse({"predict": predict}, status=200)
        # return redirect(request.META['HTTP_REFERER'])
        # return HttpResponseRedirect(request.path_info, {'predict': predict})
        # return HttpResponseRedirect("cough")


    return render(request, 'demo.html')

# def cough(request):
#     print(request.method)
#     if request.method == "POST":
#         print("lala: ", request.method)
#         print("query: ", request.FILES.get('query',None).name)
#         print("s1: ", request.FILES.get('s1',None))
#         print("s2: ", request.FILES.get('s2',None))
#         print("s3: ", request.FILES.get('s3',None))
#         print("s4: ", request.FILES.get('s4',None))
#         query = request.FILES.get('query',None).name
#         s1 = request.FILES.get('s1',None).name
#         s2 = request.FILES.get('s2',None).name
#         s3 = request.FILES.get('s3', None).name
#         s4 = request.FILES.get('s4', None).name
#         xq = getData(query).reshape(1, 1, 51, 40)
#         xs1 = getData(s1).reshape(1, 1, 51, 40)
#         xs2 = getData(s2).reshape(1, 1, 51, 40)
#         xs3 = getData(s3).reshape(1, 1, 51, 40)
#         xs4 = getData(s4).reshape(1, 1, 51, 40)
#         model_path = "D:\\HoangNgan\\DoAn\\best_model.pt"
#         model = torch.load(model_path)
#         xq_emb = model.encoder.forward(xq)
#         xs1_emb = model.encoder.forward(xs1)
#         xs2_emb = model.encoder.forward(xs2)
#         xs3_emb = model.encoder.forward(xs3)
#         xs4_emb = model.encoder.forward(xs4)
#         pro_sp1 = (xs1_emb + xs2_emb)/2
#         pro_sp2 = (xs3_emb + xs4_emb)/2
#         label_sp1 = s1.split("_")[0]
#         label_sp2 = s3.split("_")[0]
#         label = [label_sp1, label_sp2]
#         print(label)
#         dists1 = euclidean_dist(xq_emb, pro_sp1)
#         dists2 = euclidean_dist(xq_emb, pro_sp2)
#         print(dists1, dists2)
#         if dists1 < dists2:
#             predict = label[0]
#         else:
#             predict = label[1]
#         print(predict)
#
#         # return redirect(request.path_info, {'predict': predict})
#         return JsonResponse({"predict": predict}, status=200)
#         # return redirect(request.META['HTTP_REFERER'])
#         # return HttpResponseRedirect(request.path_info, {'predict': predict})
#         # return HttpResponseRedirect("cough")
#
#
#     return render(request, 'demo.html')

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    print(n, m, d)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cough_path(file):
    path = "D:\\HoangNgan\\ThucTap\\Cough_data_Tuan1\\mono"
    label = file.split("_")[0]
    # print(label)
    file_path =os.path.join(os.path.join(path, label), file)
    return file_path

def build_mfcc_extractor():
        frame_len = 40/ 1000
        stride = 20/ 1000
        sample_rate = 16000
        mfcc = torchaudio.transforms.MFCC(sample_rate = 16000,
                                        n_mfcc=40,
                                        melkwargs={
                                            'hop_length' : int(stride*sample_rate),
                                            'n_fft' : int(frame_len*sample_rate)})
        return mfcc
def load_audio(filepath):
        desired_samples = int(16000*1000/1000)
        sound, _ = torchaudio.load(filepath=filepath,
                                         num_frames=desired_samples)
#         d[out_field] = sound
        return sound
def getData(file):
    sound = load_audio(cough_path(file))
    mfcc = build_mfcc_extractor()
#     x = mfcc(sound).reshape(1,51,40)
    features = mfcc(sound)[0]
    features = features.T # f x t -> t x f
    x = torch.unsqueeze(features,0)
    # print(x.shape)
    return x

def audio(request, query, s1, s2, s3, s4):

    return render(request, 'home.html')
    # return redirect("cough")



