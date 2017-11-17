// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This program was used to train the resnet34_1000_imagenet_classifier.dnn
    network used by the dnn_imagenet_ex.cpp example program.  

    You should be familiar with dlib's DNN module before reading this example
    program.  So read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp first.  
*/



#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <iterator>
#include <thread>
#include <type_traits>

using namespace std;
using namespace dlib;
 
// ----------------------------------------------------------------------------------------

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;


template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;


// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level1 = res<512,res<512,res_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = res<256,res<256,res<256,res<256,res<256,res_down<256,SUBNET>>>>>>;
template <typename SUBNET> using level3 = res<128,res<128,res<128,res_down<128,SUBNET>>>>;
template <typename SUBNET> using level4 = res<64,res<64,res<64,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
template <typename SUBNET> using alevel3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<64,ares<64,ares<64,SUBNET>>>;

// original network type
using onet_type = loss_multiclass_log<fc<1000,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<64,7,7,2,2,
                            input_rgb_image_sized<227>
                            >>>>>>>>>>>;

// training network type
using net_type = loss_multiclass_log<fc<80,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<64,7,7,2,2,
                            input_rgb_image_sized<227>
                            >>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_multiclass_log<fc<80,avg_pool_everything<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<64,7,7,2,2,
                            input_rgb_image_sized<227>
                            >>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

rectangle make_random_cropping_rect_resnet(
    const matrix<rgb_pixel>& img,
    dlib::rand& rnd
)
{
    // figure out what rectangle we want to crop from the image
    double mins = 0.75, maxs = 0.95;
    auto scale = mins + rnd.get_random_double()*(maxs-mins);
    auto size = scale*std::min(img.nr(), img.nc());
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
    return move_rect(rect, offset);
}

// ----------------------------------------------------------------------------------------

void randomly_crop_image (
    const matrix<rgb_pixel>& img,
    matrix<rgb_pixel>& crop,
    dlib::rand& rnd
)
{
    auto rect = make_random_cropping_rect_resnet(img, rnd);

    // now crop it out as a 227x227 image.
    extract_image_chip(img, chip_details(rect, chip_dims(227,227)), crop);

    // Also randomly flip the image
    if (rnd.get_random_double() > 0.5)
        crop = fliplr(crop);

    // And then randomly adjust the colors.
    apply_random_color_offset(crop, rnd);
}

void randomly_crop_images (
    const matrix<rgb_pixel>& img,
    dlib::array<matrix<rgb_pixel>>& crops,
    dlib::rand& rnd,
    long num_crops
)
{
    std::vector<chip_details> dets;
    for (long i = 0; i < num_crops; ++i)
    {
        auto rect = make_random_cropping_rect_resnet(img, rnd);
        dets.push_back(chip_details(rect, chip_dims(227,227)));
    }

    extract_image_chips(img, dets, crops);

    for (auto&& img : crops)
    {
        // Also randomly flip the image
        if (rnd.get_random_double() > 0.5)
            img = fliplr(img);

        // And then randomly adjust the colors.
        apply_random_color_offset(img, rnd);
    }
}

dlib::matrix<rgb_pixel> randomly_rotate_image(const dlib::matrix<dlib::rgb_pixel>& in, dlib::rand& rnd) {
  dlib::matrix<dlib::rgb_pixel> out;
  auto angle = rnd.get_double_in_range(-3.0, 3.0) * dlib::pi / 180.0;
  dlib::rotate_image(in, out, angle);
  return out;
};

// ----------------------------------------------------------------------------------------

struct image_info
{
    string filename;
    long numeric_label;
};

std::vector<image_info> get_train_listing()
{
    char buf[8192];
    std::vector<image_info> results;
    FILE* fin = fopen("ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json", "r");
    rapidjson::FileReadStream in(fin, buf, sizeof(buf));
    rapidjson::Document doc;
    doc.ParseStream(in);
    for (auto i = doc.Begin(); i != doc.End(); ++i) {
      image_info t;
      t.filename = "ai_challenger_scene_train_20170904/scene_train_images_20170904/";
      t.filename += (*i)["image_id"].GetString();
      t.numeric_label = atol((*i)["label_id"].GetString());
      results.emplace_back(t);
    }
    return results;
}

std::vector<image_info> get_validation_listing()
{
    char buf[8192];
    std::vector<image_info> results;
    FILE* fin = fopen("ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json", "r");
    rapidjson::FileReadStream in(fin, buf, sizeof(buf));
    rapidjson::Document doc;
    doc.ParseStream(in);
    for (auto i = doc.Begin(); i != doc.End(); ++i) {
      image_info t;
      t.filename = "ai_challenger_scene_validation_20170908/scene_validation_images_20170908/";
      t.filename += (*i)["image_id"].GetString();
      t.numeric_label = atol((*i)["label_id"].GetString());
      results.emplace_back(t);
    }
    return results;
}

// ----------------------------------------------------------------------------------------

#define DECLARE_HAS_CLASS_MEMBER(NAME) \
  template<typename T, typename... ARGS> \
  struct has_member_##NAME { \
    template<typename U> constexpr static auto check(int)->decltype(std::declval<U>().NAME(std::declval<ARGS>()...), std::true_type()); \
    template<typename U> constexpr static std::false_type check(...); \
    static constexpr bool value = decltype(check<T>(0))::value; \
  }; \
  template<typename T> \
  struct has_member_##NAME<T, void> { \
    template<typename U> constexpr static auto check(int)->decltype(std::declval<U>().NAME(), std::true_type()); \
    template<typename U> constexpr static std::false_type check(...); \
    static constexpr bool value = decltype(check<T>(0))::value; \
  }

#define HAS_CLASS_MEMBER(CLASS, MEMBER, ...) \
  has_member_##MEMBER<CLASS, __VA_ARGS__>::value

DECLARE_HAS_CLASS_MEMBER(set_learning_rate_multiplier);
DECLARE_HAS_CLASS_MEMBER(layer_details);

template <class T>
typename std::enable_if<HAS_CLASS_MEMBER(T, set_learning_rate_multiplier, double), void>::type visit(T& layer) {
  layer.set_learning_rate_multiplier(0.0);
  layer.set_bias_learning_rate_multiplier(0.0);
}

template <class T>
typename std::enable_if<!HAS_CLASS_MEMBER(T, set_learning_rate_multiplier, double), void>::type visit(T& layer) {
  // nop
}

struct layer_visitor {
  template <class T>
  typename std::enable_if<HAS_CLASS_MEMBER(T, layer_details, void), void>::type operator()(size_t idx, T& net) {
    visit(net.layer_details());
  }

  template <class T>
  typename std::enable_if<!HAS_CLASS_MEMBER(T, layer_details, void), void>::type operator()(size_t idx, T& net) {
    // nop
  }
};

int main(int argc, char** argv) try
{
    size_t batch_size = 75;
    size_t threshold = 10000;
    if (argc >= 2) {
      batch_size = atol(argv[1]);
      if (argc >= 3) {
        threshold = atol(argv[2]);
      }
    }

    cout << "\nSCANNING IMAGENET DATASET\n" << endl;

    auto listing = get_train_listing();
    cout << "images in dataset: " << listing.size() << endl;

    set_dnn_prefer_smallest_algorithms();


    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    onet_type onet;
    deserialize("resnet34_1000_imagenet_classifier.dnn") >> onet;
    visit_layers_range<11, onet_type::num_layers>(onet, layer_visitor{});
    net_type net;
    net.subnet().subnet() = onet.subnet().subnet();

    dnn_trainer<net_type> trainer(net,sgd(weight_decay, momentum));
    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    trainer.set_synchronization_file("trainer_state_file.dat", std::chrono::minutes(10));
    // This threshold is probably excessively large.  You could likely get good results
    // with a smaller value but if you aren't in a hurry this value will surely work well.
    //trainer.set_iterations_without_progress_threshold(10000);
    trainer.set_iterations_without_progress_threshold(threshold);
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    set_all_bn_running_stats_window_sizes(net, 1000);

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<unsigned long> labels;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<std::pair<image_info,matrix<rgb_pixel>>> data(200);
    auto f = [&data, &listing](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        matrix<rgb_pixel> img;
        std::pair<image_info, matrix<rgb_pixel>> temp;
        while(data.is_enabled())
        {
            temp.first = listing[rnd.get_random_32bit_number()%listing.size()];
            load_image(img, temp.first.filename);
            randomly_crop_image(randomly_rotate_image(img, rnd), temp.second, rnd);
            data.enqueue(temp);
        }
    };
    std::thread data_loader1([f](){ f(1); });
    std::thread data_loader2([f](){ f(2); });
    std::thread data_loader3([f](){ f(3); });
    std::thread data_loader4([f](){ f(4); });

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-3.
    while(trainer.get_learning_rate() >= initial_learning_rate*1e-3)
    {
        samples.clear();
        labels.clear();

        // make a 160 image mini-batch
        std::pair<image_info, matrix<rgb_pixel>> img;
        //while(samples.size() < 160)
        while(samples.size() <= batch_size)
        {
            data.dequeue(img);

            samples.push_back(std::move(img.second));
            labels.push_back(img.first.numeric_label);
        }

        trainer.train_one_step(samples, labels);
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    data.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

    // also wait for threaded processing to stop in the trainer.
    trainer.get_net();

    net.clean();
    cout << "saving network" << endl;
    serialize("resnet34.dnn") << net;






    // Now test the network on the imagenet validation dataset.  First, make a testing
    // network with softmax as the final layer.  We don't have to do this if we just wanted
    // to test the "top1 accuracy" since the normal network outputs the class prediction.
    // But this snet object will make getting the top5 predictions easy as it directly
    // outputs the probability of each class as its final output.
    softmax<anet_type::subnet_type> snet; snet.subnet() = net.subnet();

    cout << "Testing network on imagenet validation dataset..." << endl;
    int num_right = 0;
    int num_wrong = 0;
    int num_right_top1 = 0;
    int num_wrong_top1 = 0;
    dlib::rand rnd(time(0));
    // loop over all the imagenet validation images
    for (auto l : get_validation_listing())
    {
        dlib::array<matrix<rgb_pixel>> images;
        matrix<rgb_pixel> img;
        load_image(img, l.filename);
        // Grab 16 random crops from the image.  We will run all of them through the
        // network and average the results.
        const int num_crops = 16;
        randomly_crop_images(img, images, rnd, num_crops);
        // p(i) == the probability the image contains object of class i.
        matrix<float,1,80> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;

        // check top 1 accuracy
        if (index_of_max(p) == l.numeric_label)
            ++num_right_top1;
        else
            ++num_wrong_top1;

        // check top 5 accuracy
        bool found_match = false;
        for (int k = 0; k < 3; ++k)
        {
            long predicted_label = index_of_max(p);
            p(predicted_label) = 0;
            if (predicted_label == l.numeric_label)
            {
                found_match = true;
                break;
            }

        }
        if (found_match)
            ++num_right;
        else
            ++num_wrong;
    }
    cout << "val top3 accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;
    cout << "val top1 accuracy:  " << num_right_top1/(double)(num_right_top1+num_wrong_top1) << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

