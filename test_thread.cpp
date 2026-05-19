#include <thread>
#include <mutex>
#include <atomic>
int main(){
    std::thread t([](){});
    t.join();
    std::mutex m;
    std::atomic<int> a{0};
    return 0;
}
