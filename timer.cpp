#include "Timer.h"

// Constructor
Timer::Timer()
    : time_elapsed(0.0f) {
    newt = std::chrono::high_resolution_clock::now();
    oldt = newt;
}

// Start the timer
void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    time_elapsed = 0.0f;
    previous_times.clear();
}

// Update the timer
void Timer::update() {
    oldt = newt;
    newt = std::chrono::high_resolution_clock::now();

    // Calculate time elapsed since the start
    time_elapsed = std::chrono::duration<float>(newt - start_time).count();

    // Record frame time
    double frame_time = std::chrono::duration<float>(newt - oldt).count();
    if (previous_times.size() >= ESTIMATE_FROM_LAST_FRAMES) {
        previous_times.erase(previous_times.begin()); // Remove oldest frame time
    }
    previous_times.push_back(frame_time);
}

// Get the total time elapsed since the timer started
float Timer::getTimeElapsed() const {
    return time_elapsed;
}

// Get the vector of previous frame times
const std::vector<float>& Timer::getPreviousTimes() const {
    return previous_times;
}
