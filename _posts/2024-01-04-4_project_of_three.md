---
layout: post
category: Embedded Systems
tag: Digital Signal Processing on STM32F4
---

# 3 Projects: From Complex To Much Complex

---
_table of contents_
1. TOC
{:toc}
---

I am writing these posts at the end of the project [1]. On the next pages, there should be 3 projects with different approaches. 

I will describe the problems I faced during the construction of these three projects and how I solved them. While creating this series, I noticed that people struggle to process even the simplest digitla signals in the STM32 environment due to limited resources. That's why we aimed to keep the projects simple and easy to understand. Unfortunately, in embedded systems, what seems simple often comes with a bunch of technical challenges behind the scenes.

The three projects are chronologically listed as follows:

1. Baremetal UART Approach
2. Baremetal USB Approach
3. RTOS, USB Approach with CLI (most completed one)


## Summary of Baremetal UART
The first project tries to find an answer for the question: "What can i do with only using HAL library, UART, DMA and math.h?". The first project aims to focus on digital signal processing while keeping the exposure to embedded systems as low as possible. This approach enables users to explore this domain while successfully completing meaningful projects. Regrettably, as one might infer, the genuine challenge lies in achieving "**real-time** digital signal processing."

## Why to not use UART

UART, renowned for its longevity and user-friendly nature since World War II, has drawbacks in contemporary applications. First of all, it is very slow to send audio signals without hearing jitters and noises. As you will see on the first project, if you are generating a simple cosinus signal with floats, you cannot even send them in time and as a result we miss the deadline of the packet and the disillusion of continuity disappears.

Second, as you may see on the pages, some people asks how to capture uart packets on the PC side. Due to the absence of a synchronous time clock with the serial port, UART sends packets at different intervals. If the receiving software fails to synchronize with these diverse transmission times, there is a considerable risk of misinterpreting data.


## Why to not use Baremetal
As you will see on the second project, the main loop `while(1) {// here there are functions we call to operate communications}` contains different functions handling USB protocol, signal generation and filtering tasks. The only solution To manage these functions, I used headers that lock functions into a while loop, waiting for tasks to complete or for DMA to signal flags. However, this waiting approach, involving repetitive checks, leads to significant power consumption and introduces jitters. 

This approach actually can be quite useful if you are dealing with simple projects. While suitable for simpler projects, this method becomes impractical when multitasking becomes crucial for efficiently utilizing limited resources, such as the 168 MHz clock cycle of the Cortex-M4 processor. Even though I managed to work on Baremetal, this approach is not feasible as the complexity of the project increased. If you try to add more complexity, soon enough there would be spaghetti code and broken processes everywhere. What we truly need is a more abstract and streamlined approach.


## Projects Continue
If you are ready, be intimidated by my ugliest code on UART. These were my one of the first attemps to create a working prototype. But the idea behind this is very simple and I will try to explain them. 


[1] kind of finished. I am currently writing in a hurry! Anyway...