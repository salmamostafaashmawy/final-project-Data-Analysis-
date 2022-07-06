
let upload=document.getElementById('uploadconfirm').addEventListener('click',()=>{
     location.assign('http://127.0.0.1:5000/plot');
    let values =[]
    let heightScale
    let xScale
    let xAxisScale
    let yAxisScale
    let xAxis
    let yAxis

    let width = 800
    let height = 600
    let padding = 40

    Papa.parse(document.getElementById('uploadfile').files[0],
    {
        download:true,
        header:true,
        skipEmptyLines:true,
        complete:(results)=>{
          
            for(i=0;i<results.data.length;i++){
            
                values.push(Object.values(results.data[i]))
                
            }
        console.log(values)

        var select=Object.keys(results.data[0]);
        var option=""
        for(i in select){
            option+= '<option value="' +select[i]+'">'+ select[i]+"</option>"
        }
        document.getElementById('x_axis').innerHTML+=option;
        document.getElementById('y_axis').innerHTML+=option;
        
        document.getElementById('select').addEventListener('click',()=>{
            
            final_x_value=select.indexOf(selectedX)
            final_y_value=select.indexOf(selectedY)

            let svg = d3.select('body')
                        .append('svg')


            let generateScales = () => {
                heightScale = d3.scaleLinear()
                                .domain([0, d3.max(values, (item) => {
                                    return item[final_y_value]
                                })])
                                .range([0, height-(2 * padding)])
                
                xScale = d3.scaleLinear()
                                .domain([0, values.length-1])
                                .range([padding, width-padding])
            
            }
            
            let drawCanvas = () => {
                svg.attr('width', width)
                svg.attr('height', height)
            }
            
            let drawBars = () => {
            
                let tooltip = d3.select('body').append('div')	
                                .attr('id', 'tooltip')
                                .attr('x',(item, index) => {
                                    return xScale(index)
                                })			
                                .style('visibility', 'hidden')
            
                svg.selectAll('rect')
                    .data(values)
                    .enter()
                    .append('rect')
                    .attr('class', 'bar')
                    .attr('height', (item) => {
                        return heightScale(item[final_y_value])
                    })
                    .attr('width', (width - (2*padding)) / values.length)
                    .attr('x', (item, index) => {
                        return xScale(index)
                    })
                    .attr('y', (item) => {
                        return (height-padding) - heightScale(item[final_y_value])
                    })
                    .attr('x_value', (item) => {
                        return item[final_x_value]
                    })
                    .attr('y_value', (item) => {
                        return item[final_y_value]
                    })
                    .append("title")
                    .text((d) =>"["+ d[final_x_value]+','+d[final_y_value]+']')
                    .on('mouseover', (item) => {		
                        tooltip.transition()
                            .style('visibility', 'visible')
                        document.querySelector('#tooltip').setAttribute('x_value', ()=>item[final_x_value])
                        document.querySelector('#tooltip').textContent = item[final_x_value]
                    })
                    .on('mouseout', (d) => {		
                        tooltip.transition()
                            .style('visibility', 'hidden')					
                    })


                    
            }
            
            let generateAxes = () => {
            
                let x_array = values.map((item) => {
                    return item[final_x_value]
                })
                xAxisScale = d3.scaleLinear()
                                    .domain([d3.min(x_array), d3.max(x_array)])
                                    .range([padding, width-padding])
            
                yAxisScale = d3.scaleLinear()
                                    .domain([0, d3.max(values, (item) => {
                                        return item[final_y_value]
                                    })])
                                    .range([height-(2 * padding), 0])
            
                xAxis = d3.axisBottom(xAxisScale)
                                    .tickFormat(d3.format('d'))
                yAxis = d3.axisLeft(yAxisScale)
            
                svg.append('g')
                    .call(xAxis)
                    .attr('id', 'x-axis')
                    .attr('transform', 'translate(0, '+ (height - padding) + ')')
            
                svg.append('g')
                    .call(yAxis)
                    .attr('id', 'y-axis')
                    .attr('transform', 'translate(' + padding + ', ' + padding + ')')
            }
            generateScales()
            drawCanvas()
            generateAxes()
            drawBars()

            
            
                
            })

    }


    })

})
