---
layout: post
category: Embedded Systems
tag: Digital Signal Processing on STM32F4
---


# Simplest One: Baremetal UART Project

**_This project is on GitHub: [link](https://github.com/cylnn-dev/signal-processing-on-stm32)_**

---
_table of contents_
1. TOC
{:toc}
---

## The main Concept


Take a look at our project's main() function. First, we initialize the UART and DMA using STM32's Hardware Abstraction Library (HAL). Many development environments provide abstraction, and STM's CubeMX simplifies configurations with a user-friendly GUI, though it may present some challenges in future projects.

After initializing, we print a welcome message and jump to the transferSignal() function without returning. You might notice leftover code related to menu_message and other elements. These were part of a menu system allowing LED control by sending characters from a PC to the board. I initially used them for testing UART TX/RX functionality. Feel free to check it out, but I removed them to focus on signal processing.

	{% highlight c %}
  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
  printWelcomeMessage();
  transferSignal();


  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
   {% endhighlight %}


## Project Structure

There are 3 main tasks, namely `transferSignal()`, `generateSignal()` and `UART DMA callbacks`.

Their roles are:

- `transferSignal()` -->  Initiates the signal transfer process.

This function calls generateHeaders(uint8_t *buf) to generate the header of the UART packets and then generates the signal itself with specified frequencies and lastly calls DMA to load them on the shift registers. The inclusion of macros allows for easy control over whether to filter data or not.

	{% highlight c %}
void transferSignal() {
    generateHeaders(txBuffer);

#if (USE_FIR == 1)
    generateHeaders(filterBuffer);
#endif

    generateSignal();
    generateSignal();
#if (USE_FIR == 1)
    HAL_UART_Transmit_DMA(&huart2, &filterBuffer[0], TX_BUFFER_SIZE);
#else
    HAL_UART_Transmit_DMA(&huart2, &txBuffer[0], TX_BUFFER_SIZE);
#endif
}
   {% endhighlight %}


As you may read before, the UART is an asynchronous communication protocol lacking a synchronized clock. This can lead to occasional issues with signal capture software on the PC during tests. Misinterpretation of the 4 bytes may occur as the PC listens to the packets, potentially capturing a portion of one byte while awaiting the next. To address this, two headers are added to align the bytes and facilitate correct reading. While this simple approach neglects noise on the line, more advanced protocols can be developed by incorporating checksums and timestamps, especially when dealing with the high data rate of 115200 bytes per second.

Now, let's see the packet structure used in this context.

### Ping Pong Buffering

You may also ask why are we using two headers? Isn't it enough to have only header per packet?

Yes, the system utilizes a single buffer named txBuffer to manage two UART packets.. We are calling DMA whenever one side is completed, while we are generating the other side. This way, we are speeding up the transfer. This ping-pong buffering is often implemented with pointers. But we are doing with indexing because I was also learning on the way. Yes, it looks intimidating but bear with me. It works!


        +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
          FF | FF | FF | FF |...signal bytes... | FF | FF | FF | FF | ...signal bytes...|
        +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
        
    	per sample: 4 bytes, per header: 4 bytes, too. Samples are 32-bit floats. Each square is 1 byte


We are using floats per sample, which means 4 bytes per sample. Also we are using 4 bytes of `0xFF` for headers. The header should be something unique to differentiate from the rest of the packet, so choosing `0xFFFFFFFF` can be a good option while it means `NaN` according to [1].

### Signal Generation

See the file usart.c. These macros controls how to generate packets. In this configuration, the system transmits floats, which occupy 4 bytes each. Each packet includes 1024 samples, and the packet size is determined by adding the size of these samples to the size of two headers, each consisting of four 0xFF bytes. It's worth noting that the sampling rate can be modified according to preferences. Feel free to adjust the sampling rate as needed.

{% highlight c %}

#define BYTE_PER_SAMPLE 4
#define SAMPLE_PER_PACKET 1024
#define TX_BUFFER_SIZE (BYTE_PER_SAMPLE * (SAMPLE_PER_PACKET + 2))  // two header for two half of the buffer
#define SAMPLING_RATE 10000  // in Hz
{% endhighlight %}

Before generating signals, the `generateHeaders(uint8_t *buf)` function is used to fill the header sections. This function populates the first header and then moves to the middle of the packet, filling that section as well. By doing so, it ensures proper initialization of the headers before the signal generation process begins.

{% highlight c %}
void generateHeaders(uint8_t *buf) {
    for (int i = 0; i < BYTE_PER_SAMPLE; ++i) {
        buf[i] = 0xFF;
    }
    for (int i = (TX_BUFFER_SIZE / 2); i < (TX_BUFFER_SIZE / 2) + BYTE_PER_SAMPLE; ++i) {
        buf[i] = 0xFF;
    }
}

{% endhighlight %}

The `generateSignal()` function is responsible to fill up the packet without changing the headers. If `USE_FIR == 1`, then it should also call `applyFIR(const int degree, const float *coefs)`.

{% highlight c %}
uint8_t generateSignal() {
    for (int i = (generateFirstHalf ? (0 + BYTE_PER_SAMPLE) : (TX_BUFFER_SIZE / 2) + BYTE_PER_SAMPLE);
    i < (generateFirstHalf ? ((TX_BUFFER_SIZE / 2) - BYTE_PER_SAMPLE) : TX_BUFFER_SIZE);
    i += BYTE_PER_SAMPLE) {
        float sum = 0;
        for (int j = 0; j < N_FREQS; ++j) {
            sum += cosf(2 * M_PI * signalFreqs[j] * t);
        }
        sum *= signalAmp;

        uint8_t *sumPtr = &sum;
        for (size_t j = 0; j < BYTE_PER_SAMPLE; ++j) {
            txBuffer[i + j] = *(sumPtr + j);
        }

        t += samplingPeriod;
    }

#if (USE_FIR == 1)
    applyFIR(degree, coefs);
#endif

    TOGGLE_CHAR(generateFirstHalf);
    return generateFirstHalf;
}
{% endhighlight %}


The first for loop, `for (int i = (generateFirstHalf ? (0 + BYTE_PER_SAMPLE) : (TX_BUFFER_SIZE / 2) + BYTE_PER_SAMPLE)`, iterates through the packet. The boolean-like parameter generateFirstHalf determines which part to fill, and the loop adjusts the indexes accordingly. If generateFirstHalf is true, it starts from the beginning of the packet; otherwise, it starts from the middle plus the size of a sample.

The second for loop iterates through frequencies defined earlier. Users can input different frequencies to generate a **multi-frequency** signal.

Finally, the `TOGGLE_CHAR()` macro switches the value of generateFirstHalf using the expression #define TOGGLE_CHAR(x) (x = !x). This macro effectively toggles the boolean-like parameter to alternate between filling the first half and the second half of the packet during subsequent iterations.

---
**WARNING**

Using floats for signals is generally not a good idea, especially in the embedded systems, especially those not designed for FPGA's and Linux kernels. For applications outside of these environments, it is advisable to stick with 16-bit implementations. Even if a Floating-Point Unit (FPU) is available, the stringent timing requirements can be challenging to meet. Opting for 16-bit implementations is generally more feasible and aligns better with the constraints of embedded systems.

---

`applyFIR(const int degree, const float *coefs)` uses the same for loop to iterate through every sample and applies coefficients one by one. Additionally, the last for loop iterates through one sample, which is 4 byte for our case and writes it to the packet byte by byte. This approach might seem unconventional, and considering the use of floats, it may be more straightforward to use `(uint8_t *) txBuffer`. This typecast allows direct access to the memory locations, simplifying the process and potentially enhancing clarity. Maybe I will use it next time.


{% highlight c %}
void applyFIR(const int degree, const float *coefs) {
    float *fPtr = NULL;
    uint8_t *resultPtr = NULL;
    for (int i = (generateFirstHalf ? (0 + BYTE_PER_SAMPLE) : (TX_BUFFER_SIZE / 2) + BYTE_PER_SAMPLE);
    i < (generateFirstHalf ? ((TX_BUFFER_SIZE / 2) - BYTE_PER_SAMPLE) : TX_BUFFER_SIZE);
    i += BYTE_PER_SAMPLE) {
        fPtr = &txBuffer[i];
        float sum = 0;
        for (int j = 0; j < degree; ++j) {
            if ((i / 4) - j >= 0) {  // i/4 is the index of the array, if the array type is float
                sum += coefs[j] * *(fPtr - j);
            }
        }

        // lastly, write the result to the buffer, check also if buffer addr. is given to the DMA
        resultPtr = &sum;
        for (size_t j = 0; j < BYTE_PER_SAMPLE; ++j) {
            filterBuffer[i + j] = *(resultPtr + j);
        }
    }
}
{% endhighlight %}

### DMA Callbacks

{% highlight c %}

void HAL_UART_TxHalfCpltCallback(UART_HandleTypeDef *huart) {
    while (generateFirstHalf != 0x01U) { HAL_UART_DMAPause(huart); }
    HAL_UART_DMAResume(huart);
    generateSignal();
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
    while (generateFirstHalf != 0x00U) { HAL_UART_DMAPause(huart); }
    HAL_UART_DMAResume(huart);
    generateSignal();
}
{% endhighlight %}


The `HalfCpltCallback` function is triggered by interrupts upon completion of generating the first half of the packet. Subsequently, the system proceeds to generate the second part.

On the other hand, the `CpltCallback` function is invoked upon completion of the second half. In this callback, the system generates the first half again, and the control over which part to work on is managed within the `generateSignal()` function.

These functions are defined by HAL library as `__weak` which means user can define its own implementation in any translation unit.

### Legacy Functions

{% highlight c %}

// legacy UART codes

void clearBuffer(uint8_t *buffer) {
    memset(buffer, 0, TX_BUFFER_SIZE);
}

void printSerial(const char* msg) {
    clearBuffer(txBuffer);
    for (int i = 0; i < strlen(msg); ++i) {
        txBuffer[i] = msg[i];  // maybe define more complex so give errors if something happens
    }
    HAL_UART_Transmit(&huart2, txBuffer, 1024, HAL_MAX_DELAY);
    clearBuffer(txBuffer);
}

void printNumber(uint32_t number) {
    clearBuffer(txBuffer);
    sprintf(txBuffer, "%lu", number);

    HAL_UART_Transmit(&huart2, txBuffer, sizeof(txBuffer), HAL_MAX_DELAY);
}

void clearResetTerminal() {
    HAL_UART_Transmit(&huart2, (uint8_t *) init_code, strlen(init_code), HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart2, (uint8_t *) clear_code, strlen(clear_code), HAL_MAX_DELAY);
}


void printWelcomeMessage() {
    clearResetTerminal();
    printSerial(welcome_message);
}
void printMenuMessage() {
    printSerial(menu_message);
}
{% endhighlight %}

These codes use UART in polling mode, which means it waits until the transmission is complete. `clearResetTerminal()` is sending two special codes called `init_code` and `clear_code`. PuTTY is our PC-side client and can understand [ANSI escape codes][ansi_website].

Some useful escape codes I used during development

{% highlight c %}
#define init_code "\033[H\033[J"
#define clear_code "\033[2J"
#define clear_line "\33[2K\r"
#define red_code "\u001b[31m"
#define reset_color "\u001b[0m"


#define up_n(n) ("\033[" #n "A")
#define down_n(n) ("\033[" #n "B")
#define forward_n(n) ("\033[" #n "C")
#define backward_n(n) ("\033[" #n "D")
#define save_cursor "\033[s"
#define restore_cursor "\033[u"
#define erase_to_end "\033[K"
{% endhighlight %}


## Conclusion

In concluding this project, we've successfully implemented a signal generation and transmission system on the STM32 platform using UART and DMA. Despite the functional design, there's room for improvement, especially by transitioning to USB. Incorporating USB could enhance data rates and isochronous communication, expanding the system's capabilities for future applications.


[1] "IEEE Standard for Floating-Point Arithmetic," in IEEE Std 754-2019 (Revision of IEEE 754-2008) , vol., no., pp. 48-49, 22 July 2019, doi: 10.1109/IEEESTD.2019.8766229.

[ansi_website][https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html]