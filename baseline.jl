using NPZ: npzread
using Formatting: printfmt


function main()
    path = "/home/asus/Masaüstü/DL/CoPhy/cophy/CoPhy_224/blocktowerCF/3"
    bs = 100

    train_data, test_data = get_data(path);
    data = minibatch(train_data, test_data, bs);
    mse = calculate_mse(data);

    printfmt("Mean squared error is : {:.4f}\n", mse)
end

function get_data(path)
    files = readdir(path)
    num_exp = length(files)
    train_data = zeros(4,3)
    test_data = zeros(4,3)
    for file in files
        ab_path = path * "/" * file * "/ab/states.npy"
        cd_path = path * "/" * file * "/cd/states.npy"
        b_pose = npzread(ab_path)[30,1:4,1:3]
        d_pose = npzread(cd_path)[30,1:4,1:3]
        train_data = cat(train_data, b_pose, dims=3)
        test_data = cat(test_data, d_pose, dims=3)
    end

    return train_data[:,:,2:end], test_data[:,:,2:end]
end


function minibatch(train_data, test_data, bs)
    data = []
    num_batch = Int(floor(size(train_data)[3] / bs))
    for i=0:num_batch-1
        tr = train_data[:, :, (i*bs)+1 : (i+1)*bs]
        ts = test_data[:, :, (i*bs)+1 : (i+1)*bs]
        push!(data,(tr,ts))
    end
    return data
end


function calculate_mse(data)
    loss = 0
    num_obj = 3
    for i=1:length(data)
        tr, ts = data[i]
        mean_squared_difference = sum((ts-tr).^2) / size(tr)[3]
        loss += mean_squared_difference
    end
    return loss / (length(data) * num_obj)
end
