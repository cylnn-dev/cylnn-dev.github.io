---
layout: post
category: Embedded Systems
tag: Digital Signal Processing on STM32F4
---

# RTOS USB Project: The Recommended One

## General Concept

As previously discussed, bare-metal, or in this instance, super-loop approaches, played a fundamental role in adjusting the settings. They provided an opportunity to familiarize ourselves with the STM32 family and its HAL libraries, along with UART and USB protocols. In this project, all the experiences gained during the journey are utilized to raise the standard and develop a more comprehensive solution. Certain concepts have been previously addressed in other posts and will not be reiterated here for the sake of clarity.

In this project, we managed to:

* Implemented a basic Command Line Interface (CLI) using UART. It can send different menu settings, receive user inputs, and change the menu based on user choices for visual feedback. However, it's worth noting that this CLI is not designed to handle multiple inputs simultaneously. Ways to address these limitations are discussed in the following sections.

* The board now has access to a microphone and can generate its own signals. One major challenge was the time-consuming implementation of generating multiple frequency cosine signals using floats through the USB interface. We addressed this issue by employing a Look-Up Table (LUT). We mitigated this problem using LUT (Look Up Table).

* Thanks to FreeRTOS, we can now perform tasks in real-time with minimal delays and no jitters. This virtualization has allowed us to concentrate on the digital signal processing aspect of the project.

* Users can upload their own filters to the board. However, due to performance considerations, these filters need to be quantized to 16-bit integers. While MATLAB provides excellent GUI tools and well-defined functions for designing digital filters, a current drawback is that users must manually enter the filters into the source code before running the program.

* Certain aspects of the project remain unfinished due to time constraints. However, there are leftover code snippets available that can assist those who wish to continue our project journey in the future (or maybe me).



## Project Setup

This project incorporates elements from the previous two projects, along with introducing new components that will be briefly explained.

To begin, let's discuss FreeRTOS itself.

### FreeRTOS

FreeRTOS, short for Free Real-Time Operating System, is an open-source software kernel with an MIT license. It is tailored for embedded systems, offering a lightweight and configurable solution for tasks such as managing tasks themselves, inter-task communication, memory management, and other essential features crucial for efficient development.

Moreover, FreeRTOS is supported by CubeMX out-of-the-box, or at least to some extent. Depending on the versions of FreeRTOS and CubeMX, users may encounter hardfault exceptions, which can lead to system crashes or malfunctions and are sometimes challenging to resolve. Before taking any action, it is advisable to familiarize yourself with debuggers. Previously, we utilized [GDB][gdb_link], a robust debugger. However, transitioning to RTOS applications necessitates **RTOS-aware debugging**.

[Ozone][ozone_link] is a comprehensive graphical debugger designed for embedded applications, developed by SEGGER, a partner of STMicroelectronics.

SEGGER, the same company, offers another useful tool known as [SystemView][sys_view_link]. SystemView serves as a real-time event recorder, capturing the tasks of the RTOS along with messages communicated through its API. Additionally, the recorder logs the timing of tasks, facilitating straightforward performance optimization using this information.

Both tools are easy to download and should be compatible with many STM32 development boards. Before using them, SEGGER has developed firmware that operates on the on-board ST-LINK, ensuring compatibility with J-Link. If you have been using the on-board ST-LINK so far, converting to J-Link can bring benefits in terms of upload times and enhanced debugging capabilities. Consider following [these][j_link_converter] steps for the conversion process.

---
NOTE

To integrate SEGGER SystemView, a patch needs to be applied to the default FreeRTOS build. Instructions can be found at [1]. If using Clion, the IDE comes equipped with a patch tool. Otherwise, [git diff][git_diff] can be employed, and manual patching may be necessary in some cases. Before proceeding, ensure data is backed up and be prepared to take brave steps. Also, review the modified FreeRTOS build. After patching, copy the updated FreeRTOS build to the local software repository of CubeMX.

To locate the local repository on your system:

<img alt="alt"   class="center" src="/../../assets/8_local_rep.png" />

---

The book [2] is an excellent starting point for understanding RTOS. It provides detailed steps for Ozone and SystemView, making it a valuable resource. If you are not familiar with RTOS or lack a defined development environment, I highly recommend reading this book.

If everything is set and done, then let's delve into specific sections of our projects.

### File Navigation

The files that are important to look and discuss are listed here:

* filters.c/.h: These files define filters to be applied to arrays in filter_util.c. The filters are designed using the MATLAB Filter Designer tool. There are two types of filter structs named Filter and SOSFilter, which will be explained later.

* tinyusb_utils.c/.h: These files contain stock tinyusb functions to handle USB communication. The function tud_audio_tx_done_post_load_cb(...) waits to receive a StreamBuffer and writes to the internal buffers of the library using tud_audio_write((uint8_t *)usb_buffer, received_bytes);. The StreamBuffer is sent from freertos.c.

* freertos.c: This file defines the main structure for the operation of the RTOS. All tasks and some Interrupt Service Routines (ISRs) are defined here. There are two StreamBuffers to handle UART and USB tasks, each with its own StreamBuffer.

* mic_util.c/.h: These files contain two I2S callbacks fired during communication with the microphone. They notify mic_task() to process and load the data while the DMA fills up another part of the buffer. There is also an interpolate() function, not implemented and left to the user for the problems defined later on.

* signal_generators.c: This file contains comprehensive functions for signal generation processes. There are five functions: generate_cos(void), generate_saw(void), generate_triangle(void), generate_square(void), and generate_impulse(void), to generate multifrequency Cosinus, basic SAW, TRIANGLE, SQUARE, and IMPULSE TRAIN. Additionally, there are helpful utility functions to supply data for signal_menu.

* usart.c/.h: These files define UART callbacks to handle incoming/outgoing messages between the USART2 of the board and the PC's respective port, listened to by PuTTy. Additionally, menu messages and additional messages of the CLI are defined in usart.h with names welcome_message, about_message, main_menu_message, mode_menu_message, signal_menu_message, and filter_menu_message. Positions are defined to help functions alter the menu messages.

* filter_util.c/.h: These files implement FIR, IIR-DIRECT-1, IIR-DIRECT-2, and IIR-SOS-DIRECT-1 (SOS filter implementations are not stable for now; experimental and work in progress). Respective functions are named fir_filter(int16_t *output), iir_filter_direct_1(int16_t *output), iir_filter_direct_2(int16_t *output), and iir_filter_sos_direct_1(int16_t *output). There is also a simple mechanism to select at runtime which filter to use.

* freertos_util.c/.h: Initially planned to use a container for several utility functions, but due to the increasing number of utility functions, they are separated into different locations. In the future, // todo: organize this project better!!!

* tim.c: This file has a simple function called HAL_StatusTypeDef led_channel_ctrl(uint16_t channel, bool state), which controls which LED to light with user inputs.


## Working Principles

Not all of the working principles are discussed here. The notable ones are listed for project clarification.


### Tasks

As defined in [3], a real-time application utilizing an RTOS can be structured as a collection of independent tasks. Each task operates within its own context, without relying on other tasks or the RTOS scheduler. Moreover, it should be noted that only one task can be executed if the system has one CPU core/thread. Orchestrating inter-task communication is the responsibility of the scheduler, which employs preemptive scheduling.

All tasks are defined in freertos.c, and currently, there are eight tasks:


	{% highlight c %}
/* tasks */
void init_task(void *argument);
void input_control(void *argument);
void led_control(void *argument);
void uart_transmit_task(void *argument);
void generate_signal(void *argument);
void usb_task(void *argument);
void filter_task(void *argument);
void mic_task(void *argument);
	{% endhighlight %}


#### init_task()


	{% highlight c %}
void init_task(void *argument)
{
    UNUSED(argument);
    uart_transmit(clear_code);
    uart_transmit(welcome_message);
    uart_transmit(main_menu_message);
    uart_receive();

    calculate_lut();
    calculate_dc_points();
    calculate_amplitudes();
    init_filter_array();
    enable_mic_dma();

    // todo_fixed: change tasks with user
    vTaskSuspend(mic_taskHandle);

    vTaskDelete(NULL);
}
   {% endhighlight %}


The init task is called with the highest priority and deletes itself after the required initializations. Using vTaskDelete(NULL) allows this task to delete itself, and the corresponding memory can be freed during runtime.

The program greets users with the welcome_message and main_menu_message. Following that, it enters its regular loop of listening to the user and providing the necessary feedback on the PuTTY console. Functions prefixed with calculate_ are essential for generating the cosine signal and handling menu settings. Additionally, the filter array, containing filters defined in filters.c, is initialized. The mic_task is currently suspended and serves as leftover code.

#### input_control()


	{% highlight c %}
	
void input_control(void *argument)
{
    UNUSED(argument);
    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        SEGGER_SYSVIEW_PrintfHost("rx_receive: %c", uart_rx_char);

        // process the char
        state_functions[program_state](uart_rx_char);

        // at the end re-enable dma port for another char
    }
}
   {% endhighlight %}


The input control task is activated whenever a user inputs a character using the CLI. The input is awaited indefinitely using ulTaskNotifyTake(pdTRUE, portMAX_DELAY); and is then sent to the corresponding state_function.

The state_functions array contains several menu functions to process the input with the current status of the program. These statuses are stored in the program_state. With this implementation, we have achieved a [finite-state machine][fsm_link] (FSM).


	{% highlight c %}
	
typedef enum {
    MAIN_MENU   = 0,
    MODE_MENU   = 1,
    SIGNAL_MENU   = 2,
    FILTER_MENU = 3,
} Program_States;

Program_States program_state = MAIN_MENU;
	
StateFunction state_functions[] = {main_menu, mode_menu, signal_menu, filter_menu};

/* snip */

void main_menu(const char input) {
    SEGGER_SYSVIEW_PrintfHost("main_menu running");
    switch (input) {
        case '1':
            program_state = MODE_MENU;
            uart_transmit(clear_code);
            uart_transmit(mode_menu_message);
            break;
        case '2':
            program_state = SIGNAL_MENU;
            update_signal_menu();
            break;
        case '3':
            program_state = FILTER_MENU;
            update_filter_menu();
            break;
        case '9':
            uart_transmit(clear_code);
            uart_transmit(about_message);
            break;
        case '0':
            program_state = MAIN_MENU;
            uart_transmit(clear_code);
            uart_transmit(main_menu_message);
            break;
        default:
            uart_transmit(not_understood_message);
            break;
    }
    uart_receive();
}

/* mode_menu and other menu functions can be found with similar structures on freertos.c */

   {% endhighlight %}


#### led_control()


	{% highlight c %}
	
void led_control(void *argument)
{
    UNUSED(argument);
    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        SEGGER_SYSVIEW_PrintfHost("led control fired");

        led_channel_ctrl(RED_CHANNEL, (mode_states & (MIC_ENABLE | SIGNAL_ENABLE | ADC_ENABLE)));
        led_channel_ctrl(ORANGE_CHANNEL, (mode_states & (FILTER_ENABLE)));
        led_channel_ctrl(BLUE_CHANNEL, (mode_states & (DAC_ENABLE | DRIVER_ENABLE)));
        led_channel_ctrl(GREEN_CHANNEL, (mode_states & (USB_ENABLE)));
    }
}

   {% endhighlight %}


The led_control() function provides a fundamental structure for notifying whenever a mode_state change occurs. It examines the bits of the 32-bit value and illuminates or extinguishes the corresponding LEDs accordingly. These LEDs serve dual purposes, supporting both debugging and visual representation. Some LEDs and modes are intentionally left open for user implementation.


#### uart_transmit_task()


	{% highlight c %}
	
void uart_transmit_task(void *argument)
{
    UNUSED(argument);
    size_t received_bytes;
    while (1) {
        received_bytes = xStreamBufferReceive(uart_tx_bufferHandle,
                                              uart_tx_buffer,
                                              UART_TX_BUFFER_SIZE,
                                              portMAX_DELAY);
        if (received_bytes > 0) {
            HAL_UART_Transmit_DMA(&huart2, &uart_tx_buffer[0], received_bytes);
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);    // DMA TX ISR will notify if transfer is completed
            memset(uart_tx_buffer, 0, UART_TX_BUFFER_SIZE);
        }
        else {
            // todo: is sleeping necessary?
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    }
}

   {% endhighlight %}


This task manages the transmission of data to the UART console. As previously discussed, simultaneous TX and RX are not allowed and should be handled cautiously. If the line is busy, the task will wait for another 500 ms, which may seem excessive but is not very noticeable.

#### generate_signal()


	{% highlight c %}
	
void generate_signal(void *argument) {
    UNUSED(argument);

    while (1) {
        if (mode_states & SIGNAL_ENABLE) {
            signal_functions[signal_settings.signal_type]();
        } else if (mode_states & MIC_ENABLE) {
            uint16_t *pdm_ptr = &mic_pdm_buffer[mic_pdm_buffer_state];
            PDM_Filter((uint8_t*) pdm_ptr, signal_buffer, &PDM1_filter_handler);
        }

        /* run filter task, if it is suspended pass it */
        if (mode_states & FILTER_ENABLE) {
            xTaskNotifyGive(filter_taskHandle);
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        }

        /* send signal buffer to usb_task */
        xStreamBufferSend(usb_bufferHandle,
                          signal_buffer,
                          SIGNAL_SIZE_BYTES,
                          portMAX_DELAY);
    }
}

   {% endhighlight %}


The first if statement, if (mode_states & SIGNAL_ENABLE), determines whether to load the signal_buffer with a generated signal or microphone output. This if-else condition can be efficiently addressed for performance optimization by employing a jump table.

---
NOTE

The StateFunction state_functions[] = {main_menu, mode_menu, signal_menu, filter_menu}; serves as a jump table. While one could use a basic switch structure, there is an ongoing debate on this topic, easily accessible on the internet. This implementation is designed for improved code readability.

---

#### usb_task()


	{% highlight c %}
	
void usb_task(void *argument) {
    UNUSED(argument);

    usb_init();
    while (1) {
        tud_task();
//        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

   {% endhighlight %}


Nothing to worry about, as the TinyUSB library takes care of encoding and other tasks. The packages are scheduled every 1 ms for 48 samples at a rate of 48 kHz. It's crucial to assign a high priority to the usb_task() to avoid loss of scheduling, as any interruptions may lead to detectable noises and popups by the human ear. Ensure that the data is loaded into the tinyusb_utils.c as demonstrated earlier.

The StreamBuffer is employed for USB communication. This First-In-Last-Out (FIFO) ring buffer structure can hold up to 10 packets in cases where the CPU is too busy to generate or receive signals or process them promptly.

Consideration can also be given to using a Queue with mutex protection. This approach allows the programmer to enable mic_task() and separate these tasks without concerns about multiple readers. The Queue implementation is specifically designed for situations where exclusive access to the queue is necessary to prevent conflicts between tasks. However, if speed is crucial, the StreamBuffer is more lightweight, resulting in faster access to the elements. In USB transmission, speed is essential to allow other parts of the code to gain access to CPU time. While StreamBuffer is chosen for its speed, this method has its own set of challenges.

The official documentation of FreeRTOS is highly informative on topics of this nature, and you can find relevant information at [4].

#### filter_task()


	{% highlight c %}
	
void filter_task(void *argument) {
    UNUSED(argument);

    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY); // wait until signal buffer is ready
        memcpy(filter_buffer.current_inputs, signal_buffer, SIGNAL_SIZE_BYTES);

        if (use_sos)    iir_filter_sos_direct_1(signal_buffer);
        else   filter_functions[current_filter.filter_type](signal_buffer);

        xTaskNotifyGive(generate_signal_taskHandle);
    }
}

   {% endhighlight %}


filter_task() is responsible for invoking the appropriate function for the current filter, stored either in current_filter or current_sos_filter. The boolean use_sos is employed to determine which pointer to consider and is checked during runtime in filter_task().

### Filter Functions

Currently, there are four filter functions ready to be called by filter_task(). The implementations of these functions are as follows:


	{% highlight c %}
void fir_filter(int16_t *output) {
    for (int i = 0; i < SIGNAL_SIZE; ++i) {
        int sample = 0;

        for (int j = 0; j < current_filter.filter_size; ++j) {
            int index = i - j;
            if (index < 0) {
                index += SIGNAL_SIZE;
                sample += current_filter.numerators[j] * filter_buffer.past_inputs[index];
            } else {
                sample += current_filter.numerators[j] * filter_buffer.current_inputs[index];
            }
        }
        filter_buffer.current_outputs[i] = (int16_t) (sample >> current_filter.fraction_len);
    }

    memcpy(output, filter_buffer.current_outputs, SIGNAL_SIZE_BYTES);
    memcpy(filter_buffer.past_inputs, filter_buffer.current_inputs, SIGNAL_SIZE_BYTES);
}

void iir_filter_direct_1(int16_t *output) {
    for (int i = 0; i < SIGNAL_SIZE; ++i) {
        int32_t sample = 0;

        /* b coefficients are here */
        for (int j = 0; j < current_filter.filter_size; ++j) {
            int index = i - j;
            if (index < 0) {
                index += SIGNAL_SIZE;
                sample += current_filter.numerators[j] * filter_buffer.past_inputs[index];
            } else {
                sample += current_filter.numerators[j] * filter_buffer.current_inputs[index];
            }
        }

        /* a coefficients are here */
        for (int j = 1; j < current_filter.filter_size; ++j) {
            int index = i - j;
            if (index < 0) {
                index += SIGNAL_SIZE;
                sample -= current_filter.denominators[j] * filter_buffer.past_outputs[index];
            } else {
                sample -= current_filter.denominators[j] * filter_buffer.current_outputs[index];
            }
        }

        filter_buffer.current_outputs[i] = (int16_t) (sample >> current_filter.fraction_len);
    }

    memcpy(output, filter_buffer.current_outputs, SIGNAL_SIZE_BYTES);
    memcpy(filter_buffer.past_outputs, filter_buffer.current_outputs, SIGNAL_SIZE_BYTES);
    memcpy(filter_buffer.past_inputs, filter_buffer.current_inputs, SIGNAL_SIZE_BYTES);
}


void iir_filter_direct_2(int16_t *output) {
    for (int i = 0; i < SIGNAL_SIZE; ++i) {
        int32_t current_w_value = filter_buffer.current_inputs[i];

        /* a */
        for (int j = 1; j < current_filter.filter_size; ++j) {
            int index = i - j;
            if (index < 0) {
                index += SIGNAL_SIZE;
                current_w_value -= current_filter.denominators[j] * filter_buffer.past_w_values[index];
            } else {
                current_w_value -= current_filter.denominators[j] * filter_buffer.current_w_values[index];
            }
        }
        filter_buffer.current_w_values[i] = (int16_t) (current_w_value >> current_filter.fraction_len);

        /* b */
        int64_t sample = 0;
        for (int j = 0; j < current_filter.filter_size; ++j) {
            int index = i - j;
            if (index < 0) {
                index += SIGNAL_SIZE;
                sample += current_filter.numerators[j] * filter_buffer.past_w_values[index];
            } else {
                sample += current_filter.numerators[j] * filter_buffer.current_w_values[index];
            }
        }

        filter_buffer.current_outputs[i] = (int16_t) (sample);
    }

    memcpy(output, filter_buffer.current_outputs, SIGNAL_SIZE_BYTES);
    memcpy(filter_buffer.past_outputs, filter_buffer.current_outputs, SIGNAL_SIZE_BYTES);
    memcpy(filter_buffer.past_inputs, filter_buffer.current_inputs, SIGNAL_SIZE_BYTES);
    memcpy(filter_buffer.past_w_values, filter_buffer.current_w_values, SIGNAL_SIZE_BYTES);
}


void iir_filter_sos_direct_1(int16_t *output) {
    for (int section = 0; section < current_sos_filter.n_section; ++section) {
        for (int i = 0; i < SIGNAL_SIZE; ++i) {
            int32_t  sample = 0;

            /* b coefficients are here */
            for (int j = 0; j < current_filter.filter_size; ++j) {
                int index = i - j;
                if (index < 0) {
                    index += SIGNAL_SIZE;
                    sample += current_sos_filter.numerators[section][j] * filter_buffer.past_inputs[index];
                } else {
                    sample += current_sos_filter.numerators[section][j] * filter_buffer.current_inputs[index];
                }
            }

            /* a coefficients are here */
            for (int j = 1; j < current_filter.filter_size; ++j) {
                int index = i - j;
                if (index < 0) {
                    index += SIGNAL_SIZE;
                    sample -= current_sos_filter.denominators[section][j] * filter_buffer.past_outputs[index];
                } else {
                    sample -= current_sos_filter.denominators[section][j] * filter_buffer.current_outputs[index];
                }
            }

            filter_buffer.current_outputs[i] = (int16_t) (sample >> current_filter.fraction_len);
        }
        memcpy(filter_buffer.past_outputs, filter_buffer.current_outputs, SIGNAL_SIZE_BYTES);
        memcpy(filter_buffer.past_inputs, filter_buffer.current_inputs, SIGNAL_SIZE_BYTES);
    }

    memcpy(filter_buffer.past_inputs, filter_buffer.current_inputs, SIGNAL_SIZE_BYTES);
}
   {% endhighlight %}


These functions commonly use filter_buffer, which is a struct that holds several buffers alltogether and defined as follows:


	{% highlight c %}
	
typedef struct {
    int16_t current_inputs[SIGNAL_SIZE];
    int16_t past_inputs[SIGNAL_SIZE];
    int16_t current_outputs[SIGNAL_SIZE];
    int16_t past_outputs[SIGNAL_SIZE];
    // State variables for direct form II
    int16_t current_w_values[SIGNAL_SIZE];
    int16_t past_w_values[SIGNAL_SIZE];
} SignalRecordBuffer;
   {% endhighlight %}


Basically, the current_inputs buffer is filled before branching to one of the filter functions in filter_task().


	{% highlight c %}
void filter_task(void *argument) {
    UNUSED(argument);

    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY); // wait until signal buffer is ready
        memcpy(filter_buffer.current_inputs, signal_buffer, SIGNAL_SIZE_BYTES); // <-- here

        if (use_sos)    iir_filter_sos_direct_1(signal_buffer);
        else   filter_functions[current_filter.filter_type](signal_buffer);

        xTaskNotifyGive(generate_signal_taskHandle);
    }
}
   {% endhighlight %}


Following the execution of the filter functions, the past_inputs filter is populated with the current input values. This ensures that the next time filter_task() calls these functions, there will be space for new inputs. The functions additionally record the current and past outputs based on the implementation and filter type. The current_outputs are also passed to the output buffer, a common parameter for all filter functions. In our case, this output buffer is the signal_buffer, which will be sent to the USB stream.

---
**WARNING**

It seems there might be an issue with the SOS (Second-Order Sections) implementation, leading to potentially inaccurate results. It's advisable to thoroughly review and refine the implementation to address the observed discrepancies. Consider revisiting and validating the SOS implementation to ensure its correctness and accuracy.

---

## Designing Digital Filters

In this project, our focus was on implementing digital filters rather than designing them from scratch. For the design phase, we are using the Filter Designer, which comes with the Signal Processing Toolbox in MATLAB. The user-friendly GUI provides well-defined options and offers ready-to-implement digital filters. It's essential to take into account the aspect of quantization.

The default data format in Filter Designer is double-precision floating-point (64-bit), which may be excessive for our needs. Even though our STM32F407 features its own FPU, tailored for 32-bit floats, we are transmitting audio signals as 16-bits. Consequently, quantization must be applied before proceeding with any further processing.

<img alt="alt"   class="center" src="/../../assets/8_quant_photo.png" />

	
To configure the Filter Designer for our project, select "Filter arithmetic" as Fixed-point and set the Numerator word length to 16.

Additionally, you can modify the filter implementation by going to Edit -> Convert to Single Section and/or Convert Structure in the toolbox. However, be cautious, as after quantization and structure changes, there is a risk of the IIR filter becoming unstable. The effects of quantization and structural alterations are also depicted in the accompanying plots.

Once you have selected the desired settings, navigate to Targets -> Generate C header in the toolbox. This step finalizes the filter design, allowing you to obtain the coefficients and define them in the filters.c file within the desired structs. The fraction_len variable, crucial for your design, can be extracted from the dialog box that appears after choosing Generate C header.

<img alt="alt"   class="center" src="/../../assets/8_fraction_len.png" />

## Limitations and Future Works



## Conclusion


At the beginning of the journey, I was wondering why there are very limited resources on embedded systems even for simple tasks like USB communication, implementing simple filters and so on. Then I realized, as a candidate enginneer, these tasks have their own style of thinking and everybody should face with its own "deamon". We also learned that documenting this type of work is very hard, especially this type of world requires you to learn very different but very essential parts of the embedded systems. As I move on my journey, I will have to learn too many things but at least I wanted to have some contribution on this community of engineers. 

See you on the next projects!



[gdb_link]: https://www.sourceware.org/gdb/

[sys_view_link]: https://www.segger.com/products/development-tools/systemview/

[ozone_link]: https://www.segger.com/products/development-tools/ozone-j-link-debugger/

[j_link_converter]: https://www.segger.com/products/debug-probes/j-link/models/other-j-links/st-link-on-board/

[git_diff]: https://git-scm.com/docs/git-diff

[fsm_link]: https://en.wikipedia.org/wiki/Finite-state_machine

[filter_designer]: https://www.mathworks.com/help/signal/ug/introduction-to-filter-designer.html 

[1]: https://wiki.segger.com/FreeRTOS_with_SystemView

[2]: B. Amos, Hands-On RTOS with Microcontrollers: Building real-time embedded systems using FreeRTOS, STM32 MCUs, and SEGGER debug tools. Packt Publishing, 2020

[3]: https://www.freertos.org/taskandcr.html

[4]: https://www.freertos.org/RTOS-stream-buffer-example.html



