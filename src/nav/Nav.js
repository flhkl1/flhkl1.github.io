import React, { Component } from 'react'
import {Link, useLocation} from "react-router-dom"
import astronautHelmet from "../assets/astronaut-helmet.png"
import "../styles/nav.css"

export default class Nav extends Component {
  render() {
    return <div>
        <img src = {astronautHelmet}/>
    </div>;
  }
}
