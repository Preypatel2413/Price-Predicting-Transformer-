# from model import *
# import numpy as np
# import time
# import json

# model = PPT().to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# checkpoint = torch.load('ppt_cp.pt')

# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# # print("initial model eval : ", model.eval())
# m = model.to(device)


# stime = time.time()


# with open(f'Data/data_ (507).json', 'r') as f:
#     json_data = json.load(f)

# parsed = parse_json_data(json_data)
# print(len(parsed), parsed)
# feature_vec = feature_extract(parsed)
# print(feature_vec)

# data = normalize(feature_vec)
# context = (torch.tensor([data[:30]], dtype=torch.float)).to(device)


# prd = m.generate(context, max_new_prediction=30)

# predict = np.array(prd[0].tolist())

# predict = predict*np.std(feature_vec,axis=0) + np.mean(feature_vec, axis=0)

# a = 0
# x = []
# y = []
# for i in range(len(predict)):
#     print(i)
#     print("---->",(predict[i][:7]).tolist())
#     print("====>",(feature_vec[i][:7]).tolist())
#     x.append(abs(predict[i][5] - feature_vec[i][5]))

#     y.append(1 if( (feature_vec[30][5] - predict[i][5]) * (feature_vec[30][5] - feature_vec[i][5])) else -1)
#     a += (abs(predict[i][5] - feature_vec[i][5])**2)

# print((a/60)**0.5, a/60)
# print(x)
# print(y)
# print(time.time()-stime)





from model import *
import numpy as np
import time
import json
import matplotlib.pyplot as plt
model = PPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


checkpoint = torch.load('ppt_cp.pt')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# print("initial model eval : ", model.eval())
m = model.to(device)


stime = time.time()


with open(f'Data/data_ (500).json', 'r') as f:
        json_data = json.load(f)

parsed = parse_json_data(json_data)
feature_vec = feature_extract(parsed)

data = normalize(feature_vec)


curr = []
n_1 = []
n_2 = []
n_5 = []
n_10 = []
n_30 = []


# context = (torch.tensor([data[:30]], dtype=torch.float)).to(device)
# prd = m.generate(context, max_new_prediction=30)
# predict = np.array(prd[0].tolist())
# predict = predict*np.std(feature_vec,axis=0) + np.mean(feature_vec, axis=0)

for i in range(1,len(feature_vec)):
    context = (torch.tensor([data[:i]], dtype=torch.float)).to(device)
    prd = m.generate(context, max_new_prediction=31)
    predict = np.array(prd[0].tolist())
    predict = predict*np.std(feature_vec,axis=0) + np.mean(feature_vec, axis=0)

    # print(predict)
    curr.append(feature_vec[i][5])
    n_1.append(predict[i+1][5])
    n_2.append(predict[i+2][5])
    n_5.append(predict[i+5][5])
    n_10.append(predict[i+10][5])
    n_30.append(predict[i+30][5])

print(curr)
print(n_1)
print(n_2)
print(n_5)
print(n_10)
print(n_30)

plt.plot(curr, label='Current')
plt.plot(n_1, label='Next 1')
plt.plot(n_2, label='Next 2')
plt.plot(n_5, label='Next 5')
plt.plot(n_10, label='Next 10')
plt.plot(n_30, label='Next 30')

plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.show()

print(time.time()-stime)