---
layout: post
title:  "Hello, World!"
date:   2023-11-02 15:35:57 +0300
category: Embedded Systems
---

# Hello, World!

Here we will, _finally_, generate the first project: **Blinking a LED**. This project is essentially the _Hello, World!_ of the embedded systems since, in the embedded domain, there is nowhere to write _Hello_.

## Write the First Program

If you've set up your first project with default peripheral initialization, navigate `Your Project Name > Core > Src > main.c`. Within the while loop in that file, add the following code:

    {% highlight c %}
    
    /* Infinite loop */
    /* USER CODE BEGIN WHILE */
    while (1)
    {
    /* USER CODE END WHILE */
    
    /* USER CODE BEGIN 3 */
      HAL_GPIO_TogglePin(LD4_GPIO_Port, LD4_Pin);
      HAL_Delay(500);
    }
    /* USER CODE END 3 */
    
    {% endhighlight %}


---
**NOTE**

Make sure to insert your code between /* USER CODE BEGIN 3 */ and /* USER CODE END 3 */ sections. Neglecting this 
step might result in your code disappearing when you reconfigure the project using CubeMX. Always use the defined 
user code sections to safeguard your custom code.

---

## Flash the MCU
Before running your code on the MCU, it's a good practice to update your [ST-LINK][st_link] firmware. Follow these steps

1. Connect your device to the computer
2. In CubeIDE, go to `Help > ST-LINK` Upgrade
3. Refresh the list and, if any updates are available, proceed to upgrade your firmware.

**Finally, you can know flash your MCU.**

Upon completing this step, one of your LEDs should be flashing.

From this point forward, I won't provide detailed explanations for every step. It's time to dive into other projects


[st_link]: https://www.st.com/en/development-tools/st-link-v2.html