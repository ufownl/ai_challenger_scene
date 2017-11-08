CXX				= g++
LD				= g++
CXXFLAGS	= -std=c++11 -Wall -O3 -mavx -ftemplate-depth=1024
LDFLAGS		= -ldlib -lblas -llapack -pthread
SOURCES		= $(wildcard *.cpp)
TARGETS		= scene_trainer_resnet34 scene_predictor_resnet34 scene_trainer_resnet101 scene_predictor_resnet101 scene_trainer_resnet152 scene_predictor_resnet152

all: $(TARGETS)

%: %.o
	$(LD) $^ $(LDFLAGS) -o $@

-include $(SOURCES:.cpp=.d)

%.d: %.cpp
	$(CXX) -M $(CXXFLAGS) $^ > $@

clean:
	$(RM) *.o
	$(RM) *.d
	$(RM) $(TARGETS)

.PHONY: all clean
